import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import time
from shapely.geometry import Polygon, MultiPoint, LinearRing
from datetime import timedelta
from scipy.spatial.distance import pdist, squareform
from wdbscan import *
from tqdm.notebook import tqdm as log_progress
from sklearn.neighbors import KernelDensity
from shapelysmooth import  taubin_smooth
from skimage import measure
from shapely import make_valid, distance


def cluster_track(gdf, sr_min_sample_n, kde_offset, kde_cell_size, kde_bandwidth, kde_percentile, kde_reduction, time_log=False):
    crs_temp = "EPSG:3857"

    gdf = gdf.to_crs(crs_temp)
    
    # cluster points with wdbscan
    tic = time.time()
    pred = wdbscan_clustering(gdf)
    if time_log:
        print(f"WDbscan took: {(time.time()-tic)/60:.6f} min")

    # remove cluster that don't create a Polygon
    pred = pred.replace(pred.value_counts().index[pred.value_counts()<=2].values, pd.NA)

    # create inital stay regions geoms
    points = gdf.groupby(pred).geometry.apply(lambda x: MultiPoint(x.geometry.values)).convex_hull
    sr_geoms = gpd.GeoDataFrame(geometry=points,crs=crs_temp)
    sr_geoms = sr_geoms.loc[sr_geoms.geometry.apply(lambda x: isinstance(x,Polygon))]

    # join srs that overlap
    sr_geoms = join_overlapping_polygons(sr_geoms.geometry.to_frame())

    # assign datapoints to srs
    gdf["cluster"] = assign_points_to_polygons(gdf,sr_geoms)
    
    if kde_reduction:
        sr_geoms_new = []
        for sr_id,cluster in log_progress(gdf.groupby("cluster"),leave=False,desc="SRs"):
            if len(cluster) < sr_min_sample_n:
                sr_geoms_new.append(cluster.geometry.union_all().convex_hull)
            else:
                sr_geoms_new.extend(kde_area(cluster, kde_offset, kde_cell_size, kde_bandwidth, kde_percentile, time_log=time_log))

        sr_geoms = gpd.GeoDataFrame(geometry=sr_geoms_new,crs=crs_temp)
    
    gdf["cluster"] = assign_points_to_polygons(gdf,sr_geoms)

    gdf, sr_geoms = rm_unvalid_clusters(gdf,sr_geoms)

    # assign category to each datapoint
    gdf["cat"] = categorise_points(gdf)

    return gdf, sr_geoms

def rm_unvalid_clusters(gdf ,sr_geoms):
    cluster_valid = ((gdf.groupby("cluster").dur.sum() >= timedelta(days=2)) & (gdf.groupby("cluster").size() > 2))
    cluster_valid = list(cluster_valid.index[cluster_valid])
    cluster_invalid = list(set(gdf.cluster.dropna().unique()) - set(cluster_valid))
    gdf.cluster = gdf.cluster.replace(cluster_invalid,pd.NA)

    sr_geoms = sr_geoms.loc[cluster_valid]

    return gdf, sr_geoms

def create_polygons_with_holes(polygons):
    new_polygons = []
    used_polygons = set()

    for i, outer_polygon in enumerate(polygons):
        if i in used_polygons:
            continue
        
        holes = [
            inner_polygon.exterior.coords
            for j, inner_polygon in enumerate(polygons)
            if i != j and outer_polygon.contains(inner_polygon)
        ]

        used_polygons.update(
            j for j, inner_polygon in enumerate(polygons)
            if i != j and outer_polygon.contains(inner_polygon)
        )

        new_polygons.append(Polygon(outer_polygon.exterior.coords, holes))
    
    return new_polygons

def kde_area(cluster, offset, cell_size, bandwidth, kde_percentile, weights=None, time_log=False):
    if weights is not None:
        if weights in cluster.columns:
            w = (cluster.dur / timedelta(hours=1)).clip(upper=6)
        elif "weights" in cluster.columns:
            w = cluster.weights
        else:
            raise ValueError("Weights not found")
    else:
        w = np.ones(len(cluster))

    cluster_xy = pd.DataFrame({"lat":cluster.geometry.x, "lon":cluster.geometry.y})
    x_grid = np.arange(cluster.geometry.x.min()-offset, cluster.geometry.x.max()+offset, cell_size)
    y_grid = np.arange(cluster.geometry.y.min()-offset, cluster.geometry.y.max()+offset, cell_size)
    X,Y = np.meshgrid(x_grid, y_grid)

    xy = np.vstack([X.ravel(), Y.ravel()]).T
    xy = pd.DataFrame(xy,columns=["lat","lon"])
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)

    tic = time.time()
    kde = kde.fit(cluster_xy, sample_weight=w)
    if time_log:
        print(f"KDE fit took: {(time.time()-tic)/60:.6f} min")

    tic = time.time()
    z = np.exp(kde.score_samples(xy))
    if time_log:
        print(f"KDE score samples took: {(time.time()-tic)/60:.6f} min")

    tic = time.time()
    z = get_volume(z, X.shape)
    if time_log:
        print(f"Get volume took: {(time.time()-tic)/60:.6f} min")

    tic = time.time()
    polygons_temp = []

    contours = measure.find_contours(z, kde_percentile)
    for contour in log_progress(contours,leave=False,desc="Contours"):
        scaled_contour = np.column_stack((x_grid[contour[:, 1].astype(int)], y_grid[contour[:, 0].astype(int)]))
        if scaled_contour.shape[0] <= 2:
            continue
        linear_ring = LinearRing(scaled_contour)
        polygon = Polygon(linear_ring)
        polygon = taubin_smooth(polygon)
        if polygon.is_valid:
            polygon = make_valid(polygon)
        polygons_temp.append(polygon)

    polygons_temp = create_polygons_with_holes(polygons_temp)
    if time_log:
        print(f"Get contours took: {(time.time()-tic)/60:.6f} min")

    return polygons_temp

def get_volume(z, grid_shape):
    flat_density = z.ravel()
    sorted_indices = np.argsort(flat_density)[::-1]  # Sort in descending order
    sorted_density = flat_density[sorted_indices]
    cumulative_volume = np.cumsum(sorted_density)
    cumulative_volume /= cumulative_volume[-1] # Normalize to 1

    # Reorder cumulative volume to match original grid order
    volume_grid = np.zeros_like(flat_density)
    volume_grid[sorted_indices] = cumulative_volume
    z = volume_grid.reshape(grid_shape)
    return z

def wdbscan_clustering(gdf, epsilon=1000, mu=48):
    X = pd.DataFrame({"lat":gdf.geometry.y,"lon":gdf.geometry.x, "dur":gdf.dur}
                    ).droplevel("animal_id").reset_index()
    X["dur"] = X.dur / timedelta(hours=1)

    # create matrix for classifier
    dissimilarity_matrix = pdist(X.loc[:,["lat","lon"]], metric='euclidean')
    square_dissimilarity_matrix = squareform(dissimilarity_matrix)

    # classify
    pred = wdbscan(square_dissimilarity_matrix, weights=X.dur.values, epsilon = epsilon, mu = mu)

    pred = pd.Series(pred,index=gdf.index)
    pred = pred.replace({-1:pd.NA})

    return pred

def categorise_points(gdf):
    """Categorise points in a GPS-track into categories: stay region, excoursion, transit.

    Args:
        gdf (GeoDataFrame): GPS-track.

    Returns:
        Series: Categories.
    """
    pd.set_option('future.no_silent_downcasting', True)
    cat = pd.Series(pd.NA, index=gdf.index)
    cat.loc[gdf.cluster.notna()] = "sr"
    cat.loc[gdf.cluster.isna()] = gdf.cluster.ffill().fillna(-1).eq(gdf.cluster.bfill().fillna(-1)
                                                                      ).replace(({True: 'ex', False: 'tr'}))
    pd.reset_option('future.no_silent_downcasting')

    return cat

def join_overlapping_polygons(sr_geoms):

    intersections = sr_geoms.sjoin(sr_geoms, how='inner', predicate='intersects').reset_index().drop(columns="geometry")
    intersections.columns = ["cluster_left","cluster_right"]
    intersections = intersections.loc[intersections.cluster_right != intersections.cluster_left]
    graph = nx.from_pandas_edgelist(intersections,source="cluster_right",target="cluster_left")
    subgraphs = pd.Series(nx.connected_components(graph)).explode()

    # join overlapping polygons and override geometry
    # if no overlaps subgraphs is empty and code is skipped
    for cluster_group in subgraphs.index.unique():
        cluster_group = subgraphs.loc[cluster_group]

        # all cluster in cluster_group get id of the lowest cluster id in the group
        cluster = cluster_group.min()

        # replace geometry with union of all geometries in cluster_group
        sr_geoms.loc[cluster,"geometry"] = sr_geoms.loc[cluster_group].geometry.union_all().convex_hull

        # drop all other clusters in cluster_group
        sr_geoms = sr_geoms.drop(index=cluster_group.loc[cluster_group != cluster])

    return sr_geoms

def assign_points_to_polygons(gdf,sr_geoms):
    cluster = pd.Series(np.NAN,index=gdf.index, dtype="int64")
    for sr_id,sr_geom in zip(sr_geoms.index,sr_geoms.geometry):
        cluster.loc[gdf.within(sr_geom) | gdf.touches(sr_geom)] = sr_id
    
    return cluster