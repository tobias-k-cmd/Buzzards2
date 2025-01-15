from time import sleep
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
import os
from shapely.geometry import Point, Polygon
from tqdm.notebook import tqdm as log_progress

def load_raw_files(input_folder):
    """Load raw anitra tracking files.

    Args:
        input_folder (path): Absolute path to raw files.

    Raises:
        ValueError: Invalid files format.

    Returns:
        list of gdfs,df: List of GeoDataFrames and DataFrame with ids.
    """
    gdfs = []
    ids = []

    for filename in log_progress(os.listdir(input_folder),desc="Loading files"):
        file_path = os.path.join(input_folder, filename)

        # Load file into a DataFrame
        if filename.endswith(".csv"):
            gdf = pd.read_csv(file_path,dtype={"wrong_gps":bool},parse_dates=["datetime"],date_format='%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            raise ValueError(f"Invalid file format: {filename}. Only .csv files are supported.")
        gdf.datetime = gdf.datetime.dt.tz_localize("UTC")
        gdf.set_index(["animal_id","datetime"],inplace=True)
            
        gdf = gdf.rename(columns={"gnss_latitude":"lat",
                                "gnss_longitude":"lon",
                                "gnss_altitude_elipsoid":"alt",
                                "battery_percent":"bat", #battery.charge.percent
                                "solar_cell_percent":"solar_cell",
                                "temperature_c":"temp"})
        gdf['geometry'] = gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

        ids.append({"animal_id":gdf.reset_index().animal_id.iloc[0],"deployment_year":gdf.reset_index().datetime.min().year})
        gdfs.append(gdf)
    ids = pd.DataFrame(ids).set_index("animal_id")
    duplicates = ids.index.to_series().duplicated()
    gdfs = [gdf for gdf,duplicated in zip(gdfs,duplicates) if not duplicated]
    ids = ids.reset_index().drop_duplicates().set_index("animal_id")

    gdfs = clean_data(gdfs)

    return gdfs, ids

def clean_data(gdfs, cols="default"):
    
    if cols == "default":
        cols = ["alt","bat","solar_cell","temp","animal_code","geometry"]
    elif cols == "all":
        cols = gdfs[0].columns
    elif isinstance(cols, list):
        pass

    gdfs_new = []

    for gdf in gdfs:
        gdf = gdf.drop(gdf[gdf.wrong_gps].index)
        gdf = gdf.drop(gdf[(gdf.lat == 0) | (gdf.lon == 0)].index)
        gdf = gdf.sort_index(axis=0, level=["animal_id","datetime"])
        gdf = gdf[~gdf.index.duplicated(keep='first')]
        gdf = gdf.loc[:,cols]
        gdfs_new.append(gdf)

    return gdfs_new

def zonal_stats_for_raster(geoms,raster_path,raster_cls,raster_crs,raster_key):
    zs = zonal_stats(geoms.to_crs(raster_crs), raster_path ,categorical=True,category_map=raster_key)
    zs = pd.DataFrame(zs, index = geoms.index).fillna(0)
    zs = zs.div(zs.sum(axis=1),axis=0)
    zs_missing = set(raster_cls) - set(zs.columns)
    zs = zs.reindex(columns=zs.columns.tolist() + list(zs_missing), fill_value=0)
    return zs