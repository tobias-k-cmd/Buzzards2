import geopandas as gpd
import rasterio
import pandas as pd
from shapely import GeometryCollection, LineString, MultiLineString, Polygon
from shapelysmooth import taubin_smooth
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.ops import linemerge



def get_ecotones(regions,exact=False):
    cls_land_cover = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and Ice", 80: "Permanent water bodies", 90:"Herbaceous wetland", 100:"Moss and lichen"}
    land_cover_path = "C:/Users/Tobi/Documents/BFNME Lokal/Buzzard qgis/word land cover 2021 v2/ESA_WorldCover_10m_2021_v200_Map.tif"
    land_cover = rasterio.open(land_cover_path)
    regions = regions.to_crs("EPSG:4326")
    mask_array,mask_transform = mask(land_cover,regions, crop=True, filled=False)

    polygons = []
    categories = []
    origins = []
    for shape, value in shapes(mask_array,transform=mask_transform,connectivity=4):
        polygon = Polygon(shape["coordinates"][0], shape["coordinates"][1:])
        category = cls_land_cover[value]
        origin = regions.index[regions.intersects(polygon)][0] # statt der 0 vllt noch ne schönere Lösung finden
        polygons.append(polygon)
        categories.append(category)
        origins.append(origin)
    s = gpd.GeoDataFrame({"geometry":polygons,
                        "value":categories,
                        "origin":origins},
                        crs=land_cover.crs)
    borders = []
    for origin in s.origin.unique():
        s_sub = s[s.origin == origin]
        for i, poly1 in s_sub.iterrows():
            for j, poly2 in s_sub.iterrows():
                if i < j :
                    shared_boundary = poly1['geometry'].boundary.intersection(poly2['geometry'].boundary)

                    if not shared_boundary.is_empty:
                        if isinstance(shared_boundary, GeometryCollection):
                            shared_boundary = MultiLineString([geom for geom in shared_boundary.geoms if isinstance(geom, LineString)])
                        if isinstance(shared_boundary, MultiLineString):
                            shared_boundary = linemerge(shared_boundary)
                        if isinstance(shared_boundary, LineString):
                            shared_boundary = [shared_boundary]
                        elif isinstance(shared_boundary, MultiLineString):
                            shared_boundary = list(shared_boundary.geoms)
                        else:
                            continue
                        for line in shared_boundary:
                            if isinstance(line, LineString):
                                line = taubin_smooth(line)
                                borders.append({"geometry":line,poly1.value:True,poly2.value:True,"origin":origin})
    
    pd.set_option('future.no_silent_downcasting', True)
    borders = gpd.GeoDataFrame.from_dict(borders).fillna(False)
    pd.reset_option('future.no_silent_downcasting')

    land_cover_cls = s.value.unique()
    if exact:
        borders_new = []
        for i,group in borders.groupby("origin"):
            group = group.groupby(list(land_cover_cls),as_index=False)['geometry'].apply(lambda x: MultiLineString(x.tolist()))
            group.sort_index(axis=1, inplace=True)
            # index instead of dummies
            group.index = group.apply(lambda row: ' - '.join([col for col in group.columns.drop("geometry") if row[col]]), axis=1)
            group.drop(group.columns.drop("geometry"),axis=1,inplace=True)
            group["origin"] = i
            group["length"] = group.geometry.length
            group["len_percent"] = group["length"] / group["length"].sum()
            group["len_per_area"] = group["length"] / regions.loc[i].area
            borders_new.append(group)
        borders = pd.concat(borders_new)
        borders_wide = []
        for i,group in borders.groupby("origin"):
            df = group.drop(columns="origin").unstack().to_frame().T.swaplevel(0,1,axis=1).sort_index(axis=1)
            borders_wide.append(df)
        borders_wide = pd.concat(borders_wide,keys=borders.origin.unique()).droplevel(1)
        return borders_wide
    else:
        borders_new =[]
        for i,group in borders.groupby("origin"):
            data = []
            for land_cover in group.columns.drop(["geometry","origin"]):
                geom = MultiLineString(group.loc[group.loc[:,land_cover],"geometry"].to_list())
                data.append({"geometry":geom,"land_cover":land_cover})
            data = gpd.GeoDataFrame(data)
            data = data.set_index("land_cover")
            data["length"] = data.geometry.length
            data["len_percent"] = data["length"] / data["length"].sum()
            data["len_per_area"] = data["length"] / regions.loc[i].area
            data = data.unstack().to_frame().T.swaplevel(0,1,axis=1).sort_index(axis=1)
            # data["origin"] = i
            borders_new.append(data)
        borders_wide = pd.concat(borders_new,keys=borders.origin.unique()).droplevel(1)
        return borders_wide