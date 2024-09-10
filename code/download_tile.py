# import all necessary libraries
import os
from datetime import datetime, timedelta
import time
import requests as r
import numpy as np
import pandas as pd
import geopandas as gp
import random
import re
from tqdm import tqdm
import shutil
import calendar
import argparse

import rasterio as rio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.shutil import copy
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
from rasterio.merge import merge

import pyproj
from pyproj import Proj

from shapely.ops import transform
from shapely.geometry import Polygon

import multiprocessing
from multiprocessing.pool import ThreadPool


def get_search_params(date_str, lp_links, fieldShape):
    lp_search = [l['href'] for l in lp_links if l['rel'] == 'search'][0]    # Define the search endpoint

    # Set up a dictionary that will be used to POST requests to the search endpoint
    params = {}

    # default limit is 10, maximum limit is 250
    params['limit'] = 250

    # Defined from ROI bounds
    bbox = f'{fieldShape.bounds[0]},{fieldShape.bounds[1]},{fieldShape.bounds[2]},{fieldShape.bounds[3]}'
    params['bbox'] = bbox

    # Define start time period / end time period
    date_time = date_str + "T00:00:00Z/" + date_str + "T23:59:59Z"
    params['datetime'] = date_time

    s30_id = "HLSS30.v2.0"
    l30_id = "HLSL30.v2.0"
    params["collections"] = [s30_id, l30_id]
    # params["collections"] = [l30_id]

    return params
  
def download_unmerged_rasters(tile, day, year_start, lp_links, fieldShape):
    # Define bands to be downloaded
    l30_bands = ['B01', 'B03', 'B05', 'B06', 'B04', 'Fmask', 'B07',
        'B10', 'B02', 'B11','SAA','SZA']
    s30_bands = ['B10', 'B07', 'B09', 'B04', 'B03', 'B11', 'B06',
        'B12','B02', 'B8A', 'B08', 'B01', 'Fmask','B05','SAA','SZA']

    # Define parameters for the day
    date = year_start + timedelta(days=day)
    params = get_search_params(date.strftime('%Y-%m-%d'), lp_links, fieldShape)
    lp_search = [l['href'] for l in lp_links if l['rel'] == 'search'][0]

    s30_urls = []
    l30_urls = []

    # Search for the HLSS30 and HLSL30 items of interest:
    hls_items = r.post(lp_search, json=params).json()['features']    # Send POST request with S30 and L30 collections included
    if len(hls_items)>0:
        s_idx = []
        l_idx = []

        # find out the indices of hls_items which are from S30 and L30
        for i in range(len(hls_items)):
            h = hls_items[i]
            if h['id'].split('.')[1] == 'S30':
                s_idx.append(i)
            elif h['id'].split('.')[1] == 'L30':
                l_idx.append(i)
            else:
                continue

        # S30 rasters
        for b in s30_bands:
            for i in s_idx:
                h = hls_items[i]
                link = h['assets'][b]['href']
                if tile in link:
                    s30_urls.append(link)

        # L30 rasters
        for b in l30_bands:
            for i in l_idx:
                h = hls_items[i]
                link = h['assets'][b]['href']
                if tile in link:
                    l30_urls.append(link)
    else:
        print(f'data not available for day {day}')
        return

    if s30_urls:
        print('Sentinel-2 data available')

    for url in tqdm(s30_urls):
        save_path = os.path.join('/scratch/user/anshulya/hls/data/raw',tile,'unmerged_rasters', url.split('/')[-1].split('.')[3][:7], 'S30')
        if os.path.exists(os.path.join(save_path, url.split('/')[-1])):
            continue
        cmd = f"python /scratch/user/anshulya/hls/code/DAACDataDownload.py -dir {save_path} -f {url}"
        os.system(cmd)

    if l30_urls:
        print('Landsat data available')

    for url in tqdm(l30_urls):
        save_path = os.path.join('/scratch/user/anshulya/hls/data/raw',tile,'unmerged_rasters', url.split('/')[-1].split('.')[3][:7], 'L30')
        if os.path.exists(os.path.join(save_path, url.split('/')[-1])):
            continue
        cmd = f"python /scratch/user/anshulya/hls/code/DAACDataDownload.py -dir {save_path} -f {url}"
        os.system(cmd)

    head, tail = os.path.split(save_path)  
    print("Unmerged rasters downloaded!!! Available at {}".format(head))
    print("------------------------------------------------------------------------------------")

    return head

## Reproject downloaded rasters
def project_hls_latlon(hls, jrc, filename = 'hls_proj.tif'):
    projTIF = filename
    transform, width, height = calculate_default_transform(
        hls.crs, jrc.crs, hls.width, hls.height, *hls.bounds)
    kwargs = hls.meta.copy()
    kwargs.update({
        'crs': jrc.crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(projTIF, 'w', **kwargs) as dst:
        for i in range(1, hls.count + 1):
            reproject(
                source=rio.band(hls, i),
                destination=rio.band(dst, i),
                src_transform=hls.transform,
                src_crs=hls.crs,
                dst_transform=transform,
                dst_crs=jrc.crs,
                resampling=Resampling.nearest)
    return projTIF

def reproject_unmerged_rasters(save_path):
    print('start of reprojections')
    for filename in tqdm(os.listdir(save_path)):
        if filename == 'tile_metadata.csv':
            continue
        if os.path.isfile(os.path.join(save_path, filename)):
            jrc = rio.open('/scratch/user/anshulya/hls/data/383_LAKE_MEAD.tif')
            hls = rio.open(os.path.join(save_path, filename))
            project_hls_latlon(hls, jrc, os.path.join(save_path, filename))

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def process_hls_rasters(tile, day, year_start, lp_links, fieldShape):
    print(os.path.join('/scratch/user/anshulya/hls/data/raw',tile,'unmerged_rasters',str(year_start.year)+str(day+1).zfill(3)))
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            save_path = download_unmerged_rasters(tile, day, year_start, lp_links, fieldShape)

            if save_path == None:
                return

            if os.path.isdir(os.path.join(save_path,'L30')):
                flag1 = 1
                reproject_unmerged_rasters(os.path.join(save_path,'L30'))
            else:
                flag1 = 0

            if os.path.isdir(os.path.join(save_path,'S30')):
                flag2 = 1
                reproject_unmerged_rasters(os.path.join(save_path,'S30'))
            else:
                flag2 = 0

            return
        except Exception as e:
            print(f"Error downloading {day}: {e}. Retrying...")
            retries += 1
            time.sleep(5)  # Optional: wait for 5 seconds before retrying
    else:
        print(f"Failed to download {day} after {max_retries} attempts.")

def clip_raster(src, res, fname = 'clipped_hls.tif'):
    coords = getFeatures(res)

    # Clip the raster with Polygon
    out_image, out_transform = mask(dataset=src, shapes=coords, crop=True, nodata=-9999)

    # Update the metadata with the new bounds and resolution
    out_meta = src.meta.copy()
    out_meta.update({
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })

    with rio.open(fname, 'w', **out_meta) as dest:
        dest.write(out_image)
    return fname


def process_hls_rasters_wrapper(tile, day, year_start, lp_links, fieldShape, gids, df_):
    print(os.path.join('/scratch/user/anshulya/hls/data/raw',tile,'unmerged_rasters',str(year_start.year)+str(day+1).zfill(3)))
    process_hls_rasters(tile, day, year_start, lp_links, fieldShape)

    d = str(year_start.year)+str(day+1).zfill(3)
    if os.path.exists(os.path.join('/scratch/user/anshulya/hls/data/raw/',tile,'unmerged_rasters',d)):
        sat = os.listdir(os.path.join('/scratch/user/anshulya/hls/data/raw/',tile,'unmerged_rasters',d))[0]
        rasters = [f for f in os.listdir(os.path.join('/scratch/user/anshulya/hls/data/raw/',tile,'unmerged_rasters',d,sat))]
    
        for idx, row in df_.iterrows():
            res = df_.loc[[idx]]
            res['geometry'] = res['geometry'].buffer(0.01)
            gid = row['GRAND_ID']
            
            dir_path = os.path.join('/scratch/user/anshulya/hls/data/raw', str(gid), 'clipped_rasters', d, sat)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                for ras_f in rasters:
                    ras = rio.open(os.path.join('/scratch/user/anshulya/hls/data/raw/',tile,'unmerged_rasters',d,sat, ras_f))
                    clip_raster(ras, res, os.path.join(dir_path, ras_f))

def s2_tile_function(tile, start_date, end_date, num_workers):
    ## Get reservoir shapefile and geometry
    s2_tiles = gp.read_file('/scratch/user/anshulya/hls/data/auxiliary/gis/s2-tiles.geojson')
    fieldShape = s2_tiles.loc[s2_tiles['Name']==tile,'geometry'].values[0]

    ## Get reservoir grand IDs
    grand = gp.read_file('/scratch/user/anshulya/hls/data/auxiliary/gis/grand_conus.geojson')
    s2_res = pd.read_csv('/scratch/user/anshulya/hls/data/auxiliary/gis/sentinel_tiles.csv')
    unique_names_per_id = s2_res.groupby('GRAND_ID')['Name'].nunique()
    grand_ids_with_unique_names = unique_names_per_id[unique_names_per_id == 1].index
    filtered_df = grand[grand['GRAND_ID'].isin(grand_ids_with_unique_names)]
    for idx, row in filtered_df.iterrows():
        filtered_df.loc[idx, 's2-tile'] = s2_res.loc[s2_res['GRAND_ID']==row['GRAND_ID'], 'Name'].values[0]
    gids = list(filtered_df[filtered_df['s2-tile']=='13TDE']['GRAND_ID'])
    df_ = grand[grand['GRAND_ID'].isin(gids)]

    ## CMR-STAC setup begins here
    # Set Up Working Environment
    inDir = os.getcwd()
    os.chdir(inDir)

    stac = 'https://cmr.earthdata.nasa.gov/stac/' # CMR-STAC API Endpoint
    stac_response = r.get(stac).json()            # Call the STAC API endpoint
    for s in stac_response: print(s)

    print(f"You are now using the {stac_response['id']} API (STAC Version: {stac_response['stac_version']}). {stac_response['description']}")
    print(f"There are {len(stac_response['links'])} STAC catalogs available in CMR.")

    stac_lp = [s for s in stac_response['links'] if 'LP' in s['title']]  # Search for only LP-specific catalogs

    # LPCLOUD is the STAC catalog we will be using and exploring today
    lp_cloud = r.get([s for s in stac_lp if s['title'] == 'LPCLOUD'][0]['href']).json()
    lp_links = lp_cloud['links']

    # Go to the next LPCLOUD page which has HLS data
    # lp_ = [l['href'] for l in lp_links if l['rel'] == 'next']
    # lp_cloud = r.get(f"{lp_[0]}").json()

    for l in lp_cloud: print(f"{l}: {lp_cloud[l]}")

    lp_links = lp_cloud['links']
    for l in lp_links:
        try:
            print(f"{l['href']} is the {l['title']}")
        except:
            print(f"{l['href']}")
    ## CMR-STAC setup ends here

    year_start = datetime(start_date.year,1,1)
    day1 = start_date - year_start
    day2 = end_date - year_start
    day1 = day1.days
    day2 = day2.days
    print(day1, day2)

    ## Multiprocessing setup
    num_cores = num_workers
    print(num_cores)

    num_chunks = (day2 - day1) // num_cores

    for i in range(num_chunks):
        t1 = datetime.now()
        chunk = range(day1, day2)[i*num_cores : (i+1)*num_cores]
        with multiprocessing.Pool(processes=len(chunk)) as pool:
            results = [pool.apply_async(process_hls_rasters_wrapper, (tile, day, year_start, lp_links, fieldShape, gids, df_)) for day in chunk]

            # Wait for all processes to complete
            pool.close()
            pool.join()

            # Check for exceptions in the download process
            for result in results:
                result.get()

        t2 = datetime.now()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('URL CHUNK DONE!!!!! {}'.format(t2-t1))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    remaining_days = range(day1, day2)[num_chunks * num_cores :]
    if remaining_days:
        with multiprocessing.Pool(processes=len(remaining_days)) as pool:
            results = [pool.apply_async(process_hls_rasters, (tile, day, year_start, lp_links, fieldShape, gids, df_)) for day in remaining_days]

            # Wait for all processes to complete
            pool.close()
            pool.join()

            # Check for exceptions in the download process
            for result in results:
                result.get()

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('FULL PROCESSING DONE!!!!!')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run the HLS raster processing for a specific tile and date range.")
    
    # Add arguments for the tile, start date, and end date
    parser.add_argument("tile", type=str, help="Sentinel-2 tile identifier (e.g., '13TDE')")
    parser.add_argument("start_date", type=str, help="Start date in format YYYY-MM-DD (e.g., '2022-01-01')")
    parser.add_argument("end_date", type=str, help="End date in format YYYY-MM-DD (e.g., '2022-01-31')")
    parser.add_argument("num_workers", type=str, help="Number of workers for multiprocessing")
    
    # Parse the arguments
    args = parser.parse_args()

    # Convert the input date strings to datetime objects
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Call the s2_tile_function with the parsed arguments
        s2_tile_function(args.tile, start_date, end_date, int(args.num_workers))
    except ValueError:
        print("Error: Please provide dates in the format YYYY-MM-DD.")

    

