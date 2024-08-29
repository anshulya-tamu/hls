# import all necessary libraries
import sys
import os
from datetime import datetime, timedelta
import requests as r
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import random
# from osgeo import gdal
import re
from tqdm import tqdm
import shutil
import calendar

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

import time

from cartopy import crs
# import hvplot.xarray
# import holoviews as hv
# gv.extension('bokeh', 'matplotlib')

import multiprocessing
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor


# Function to execute another script with arguments
def lpdaac_credential_setup():
  link = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/HLS.L30.T11SQA.2021221T181517.v2.0/HLS.L30.T11SQA.2021221T181517.v2.0.B05.tif"
  cmd = "python DAACDataDownload.py -dir hls_data -f {}".format(link)
  os.system(cmd)

def preprocess_reservoir_shp(fname):
  reservoirs = gp.read_file(fname)

  reservoirs.loc[107,'name'] = 'Farmer_s Creek Reservoir'

  t = ['/' in s for s in reservoirs['name']]
  i = 0
  res_names = ['Cox Lake', 'Eagle Nest Lake_Manor Lake', 'Lake Ballinger_Lake Moonen', 'Lake Olney_Lake Cooper', 'Mustang Lake']
  for res_idx in [77, 93, 183, 249, 313]:
    reservoirs.loc[res_idx, 'name'] = res_names[i]
    i = i + 1

  i = 0
  res_names = ['Lake Gonzales','Clear Lake Wa']
  for res_idx in [218, 70]:
    reservoirs.loc[res_idx, 'name'] = res_names[i]
    i = i + 1

  for i in range(809):
    string = reservoirs.loc[i, 'name']
    regex = re.compile('[@!#$%^&*()<>?/\|}{~]')
    # Pass the string in search
    # method of regex object.
    if(regex.search(string) == None):
      continue
    else:
      print(i,string)

  return reservoirs

## Download HLS rasters
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

def download_unmerged_rasters(res_idx, day, year_start, lp_links, fieldShape):
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
        s30_urls.append(link)

    # L30 rasters
    for b in l30_bands:
      for i in l_idx:
        h = hls_items[i]
        link = h['assets'][b]['href']
        l30_urls.append(link)
  else:
    print(f'data not available for day {day}')
    return

  if s30_urls:
    print('Sentinel-2 data available')

  for url in tqdm(s30_urls):
    save_path = os.path.join(str(res_idx),'unmerged_rasters', url.split('/')[-1].split('.')[3][:7], 'S30')
    if os.path.exists(os.path.join(save_path, url.split('/')[-1])):
      continue
    cmd = f"python DAACDataDownload.py -dir {save_path} -f {url}"
    os.system(cmd)

  if l30_urls:
    print('Landsat data available')

  for url in tqdm(l30_urls):
    save_path = os.path.join(str(res_idx),'unmerged_rasters', url.split('/')[-1].split('.')[3][:7], 'L30')
    if os.path.exists(os.path.join(save_path, url.split('/')[-1])):
      continue
    cmd = f"python DAACDataDownload.py -dir {save_path} -f {url}"
    os.system(cmd)


  # Create an empty dataframe for each day
  column_names = ['id', 'tile', 'cloud_cover', 'satellite', 'no_value']
  df = pd.DataFrame(columns=column_names)

  for i, h in enumerate(hls_items):
    df.loc[i,'id'] = h['id']
    df.loc[i,'tile'] = h['id'].split('.')[2]
    df.loc[i,'cloud_cover'] = int(h['properties']['eo:cloud_cover'])
    df.loc[i,'satellite'] = h['collection']

    if h['collection'] == 'HLSS30.v2.0':
      save_path = os.path.join(str(res_idx),'unmerged_rasters', url.split('/')[-1].split('.')[3][:7], 'S30')
    elif h['collection'] == 'HLSL30.v2.0':
      save_path = os.path.join(str(res_idx),'unmerged_rasters', url.split('/')[-1].split('.')[3][:7], 'L30')
    else:
      continue

    ras = rio.open(os.path.join(save_path,h['assets']['B01']['href'].split('/')[-1]))
    ras_data = ras.read(1)
    df.loc[i,'no_value'] = ras_data[ras_data==ras.meta['nodata']].shape[0]/ras_data.flatten().shape[0]
    ras.close()

  head, tail = os.path.split(save_path)

  print('Saving dataframe...')
  df.to_csv(os.path.join(head,'tile_metadata.csv'))

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
      jrc = rio.open('383_LAKE_MEAD.tif')
      hls = rio.open(os.path.join(save_path, filename))
      project_hls_latlon(hls, jrc, os.path.join(save_path, filename))


## Merge downloaded rasters
def merge_rasters_ids(ids, bands, res_idx, save_path, folder):
  for b in tqdm(bands):
    raster_paths = [os.path.join(save_path, folder, id+'.'+b+'.tif') for id in ids]
    rasters = [rio.open(raster_path) for raster_path in raster_paths]

    # Merge the rasters into a single dataset
    b1_merged, b1_merged_transform = merge(rasters)

    # Update the metadata of the merged dataset
    b1_merged_meta = rasters[0].meta.copy()
    b1_merged_meta.update({
        'height': b1_merged.shape[1],
        'width': b1_merged.shape[2],
        'transform': b1_merged_transform
    })

    if not os.path.exists(os.path.join(str(res_idx), 'merged_rasters',save_path.split('/')[2], folder)):
      os.makedirs(os.path.join(str(res_idx), 'merged_rasters',save_path.split('/')[2], folder))

    # Create a new raster file to store the stitched data
    output_path = os.path.join(str(res_idx), 'merged_rasters', save_path.split('/')[2], folder, b +'_stitched_raster.tif')
    with rio.open(output_path, 'w', **b1_merged_meta) as dst:
        dst.write(b1_merged)

    # Close all the input rasters
    for raster in rasters:
        raster.close()


def merge_rasters(res_idx, save_path):

  print('merge starting for day {}'.format(save_path.split('/')[-1]))

  df = pd.read_csv(os.path.join(save_path,'tile_metadata.csv'))

  # Define bands to be downloaded
  l30_bands = ['B01', 'B03', 'B05', 'B06', 'B04', 'Fmask', 'B07',
        'B10', 'B02', 'B11','SAA','SZA']
  s30_bands = ['B10', 'B07', 'B09', 'B04', 'B03', 'B11', 'B06',
        'B12','B02', 'B8A', 'B08', 'B01', 'Fmask','B05','SAA','SZA']

  print('average cloud cover: {}'.format(np.mean(df['cloud_cover'])))

  for sat in df['satellite'].unique():
    if sat=='HLSS30.v2.0':
      bands = s30_bands
      folder = 'S30'
      print(sat)
      df_ = df[df['satellite']=='HLSS30.v2.0']
    elif sat=='HLSL30.v2.0':
      bands = l30_bands
      folder = 'L30'
      print(sat)
      df_ = df[df['satellite']=='HLSL30.v2.0']
    else:
      bands = None

    if not any(df_['tile'].value_counts() > 1):
      print("Overlapping tiles not available.")
      merge_rasters_ids(df_['id'], bands, res_idx, save_path, folder)

    else:
      print("Overlapping tiles available.")
      tiles = df_['tile'].unique()
      id_list = []
      for t in tiles:
        min_value = df_[df_['tile']==t]['no_value'].min()
        min_index = df_.index[(df_['tile']==t) & (df_['no_value'] == min_value)][0]
        id_list.append(df_.loc[min_index, 'id'])
      merge_rasters_ids(id_list, bands, res_idx, save_path, folder)

  print("Merging done!!! Output available at {}/merged_rasters/{}".format(res_idx, save_path.split('/')[-1]))
  print("------------------------------------------------------------------------------------")

  f = save_path.split('/')[-1]
  df.to_csv(os.path.join(str(res_idx),'merged_rasters',f,'tile_metadata.csv'))

  return f'{res_idx}/merged_rasters/{f}'


def getFeatures(gdf):
  """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
  import json
  return [json.loads(gdf.to_json())['features'][0]['geometry']]

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


def resample_jrc_raster(hls_path, jrc, filename):
  # Open the reference raster
  with rio.open(hls_path) as src:
    # Get the metadata for the reference raster
    profile = src.profile
    transform = src.transform

    # Create a new empty raster with the same shape and metadata as the reference raster
    data = np.empty(shape=(src.height, src.width), dtype=np.float32)
    new_profile = profile.copy()
    new_profile.update(dtype=data.dtype)

    # Create a transform for the new raster using the same resolution as the reference raster
    new_transform = Affine(transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f)

    # Open the source raster to be used for updating the empty raster
    with jrc as src2:
      # Update each pixel in the new raster with the nearest neighbor value from the source raster
      rio.warp.reproject(source=rio.band(src2, 1), destination=data,
                              src_transform=src2.transform, src_crs=src2.crs,
                              dst_transform=new_transform, dst_crs=src.crs,
                              resampling=rio.warp.Resampling.nearest)

    # Write the updated raster to a new file
    with rio.open(filename, 'w', **new_profile) as dst:
      dst.write(data, 1)
      return filename



def process_hls_rasters(res_idx, day, year_start, lp_links, fieldShape):
  retries = 0
  max_retries = 3
  while retries < max_retries:
    try:
      save_path = download_unmerged_rasters(res_idx, day, year_start, lp_links, fieldShape)

      # save_path = os.path.join(str(res_idx),'unmerged_rasters',str(year_start.year)+str(day).zfill(3))

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

      if flag1 | flag2:
        merge_path = merge_rasters(res_idx, save_path)
        shutil.rmtree(save_path)

      return merge_path
    except Exception as e:
      print(f"Error downloading {day}: {e}. Retrying...")
      retries += 1
      time.sleep(5)  # Optional: wait for 5 seconds before retrying
  else:
    print(f"Failed to download {day} after {max_retries} attempts.")

def clip_rasters(reservoirs, res_idx, day, year_start, lp_links, fieldShape, im_name):
  print(os.path.join(str(res_idx),'clipped_rasters',str(year_start.year)+str(day+1).zfill(3)))

  if os.path.isdir(os.path.join(str(res_idx),'clipped_rasters',str(year_start.year)+str(day+1).zfill(3))):
    if os.path.isdir(os.path.join(str(res_idx),'merged_rasters',str(year_start.year)+str(day+1).zfill(3))):
      shutil.rmtree(os.path.join(str(res_idx),'merged_rasters',str(year_start.year)+str(day+1).zfill(3)))
    if os.path.isdir(os.path.join(str(res_idx),'unmerged_rasters',str(year_start.year)+str(day+1).zfill(3))):
      shutil.rmtree(os.path.join(str(res_idx),'unmerged_rasters',str(year_start.year)+str(day+1).zfill(3)))

    print('Clipped rasters exist for day {}, moving to the next day'.format(day))
    print('----------------------------------------------------------------------------')
    return


  merge_path = process_hls_rasters(res_idx, day, year_start, lp_links, fieldShape)

  if merge_path == None:
    return

  print(merge_path)

  res = reservoirs.iloc[[res_idx]]
  res['geometry'] = res['geometry'].buffer(0.01)
  jrc = rio.open(os.path.join('jrc_data',im_name +'.tif'))

  if os.path.isdir(os.path.join(merge_path,'S30')):
    for ras_path in tqdm(os.listdir(os.path.join(merge_path,'S30'))):
      ras = rio.open(os.path.join(merge_path,'S30',ras_path))

      dir_path = os.path.join(str(res_idx), 'clipped_rasters', merge_path.split('/')[-1], 'S30')
      if not os.path.exists(dir_path):
        os.makedirs(dir_path)
      clip_raster(ras, res, os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'S30',ras_path))

    jrc = rio.open(os.path.join('jrc_data',im_name +'.tif'))
    resample_jrc_raster(os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'S30',ras_path), jrc,
                          os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'S30','jrc.tif'))
    shutil.copy(os.path.join(merge_path,'tile_metadata.csv'), os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'tile_metadata.csv'))
    shutil.rmtree(os.path.join(merge_path,'S30'))

  if os.path.isdir(os.path.join(merge_path,'L30')):
    for ras_path in tqdm(os.listdir(os.path.join(merge_path,'L30'))):
      ras = rio.open(os.path.join(merge_path,'L30',ras_path))

      dir_path = os.path.join(str(res_idx), 'clipped_rasters', merge_path.split('/')[-1], 'L30')
      if not os.path.exists(dir_path):
        os.makedirs(dir_path)
      clip_raster(ras, res, os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'L30',ras_path))

    jrc = rio.open(os.path.join('jrc_data',im_name +'.tif'))
    resample_jrc_raster(os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'L30',ras_path), jrc,
                          os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'L30','jrc.tif'))
    shutil.copy(os.path.join(merge_path,'tile_metadata.csv'), os.path.join(str(res_idx),'clipped_rasters',merge_path.split('/')[-1],'tile_metadata.csv'))
    shutil.rmtree(os.path.join(merge_path,'L30'))

  shutil.rmtree(os.path.join(merge_path))

  return

def clip_rasters_wrapper(args):
  return clip_rasters(*args)

def main_hls_function(year, days, res_idx):
  ## Get reservoir shapefile and geometry
  fname = '800_res.geojson'
  reservoirs = preprocess_reservoir_shp(fname)
  fieldShape1 = reservoirs['geometry'][res_idx] # Define the geometry as a shapely polygon
  fieldShape = fieldShape1.buffer(0.01)

  if pd.isna(reservoirs.loc[res_idx,'grand_id']) or reservoirs.loc[res_idx,'grand_id']==-999:
    im_name = str(0) + '_' + reservoirs.loc[res_idx,'name'].replace(" ", "_")
  else:
    im_name = str(int(reservoirs.loc[res_idx,'grand_id'])) + '_' + reservoirs.loc[res_idx,'name'].replace(" ", "_")

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
  lp_ = [l['href'] for l in lp_links if l['rel'] == 'next']
  lp_cloud = r.get(f"{lp_[0]}").json()

  for l in lp_cloud: print(f"{l}: {lp_cloud[l]}")

  lp_links = lp_cloud['links']
  for l in lp_links:
    try:
      print(f"{l['href']} is the {l['title']}")
    except:
      print(f"{l['href']}")
  ## CMR-STAC setup ends here

  year_start = datetime(year,1,1)
  t1 = datetime.now()

  # for day in range(days):
  #   clip_rasters(reservoirs, res_idx, day, year_start, lp_links, fieldShape)

  # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  # print('FULL PROCESSING DONE!!!!! {} seconds'.format(datetime.now()-t1))
  # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


  ## Multiprocessing setup
  num_cores = 48
  print(num_cores)

  tasks = [(reservoirs, res_idx, day, year_start, lp_links, fieldShape, im_name) for day in range(days)]

  with ProcessPoolExecutor(max_workers=num_cores) as executor:
    results = list(executor.map(clip_rasters_wrapper, tasks))

  t2 = datetime.now()
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  print('FULL PROCESSING DONE!!!!! {}'.format(t2-t1))
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

  # num_chunks = days // num_cores

  # for i in range(num_chunks):
  #   t1 = datetime.now()
  #   chunk = range(days)[i*num_cores : (i+1)*num_cores]
  #   with multiprocessing.Pool(processes=len(chunk)) as pool:
  #     results = [pool.apply_async(clip_rasters, (reservoirs, res_idx, day, year_start, lp_links, fieldShape)) for day in chunk]

  #     # Wait for all processes to complete
  #     pool.close()
  #     pool.join()

  #     # Check for exceptions in the download process
  #     for result in results:
  #       result.get()


  #   t2 = datetime.now()
  #   print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  #   print('URL CHUNK DONE!!!!! {}'.format(t2-t1))
  #   print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

  # remaining_days = range(days)[num_chunks * num_cores :]
  # if remaining_days:
  #   with multiprocessing.Pool(processes=len(remaining_days)) as pool:
  #     results = [pool.apply_async(clip_rasters, (reservoirs, res_idx, day, year_start, lp_links, fieldShape)) for day in remaining_days]

  #     # Wait for all processes to complete
  #     pool.close()
  #     pool.join()

  #     # Check for exceptions in the download process
  #     for result in results:
  #       result.get()

  #   print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  #   print('FULL PROCESSING DONE!!!!!')
  #   print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


if __name__ == "__main__":
  # Check if arguments are provided
  if len(sys.argv) >= 4:

    # LPDAAC credential setup
    lpdaac_credential_setup()

    # Get the arguments
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    try:
      arg3 = int(sys.argv[3])  # Convert the third argument to an integer
    except ValueError:
      print("Error: The third argument must be an integer.")
      sys.exit(1)

    # Process datetime arguments and create datetime objects
    try:
      start_date = datetime.strptime(arg1, "%d-%m-%Y")
      end_date = datetime.strptime(arg2, "%d-%m-%Y")
    except ValueError:
      print("Error: Invalid date format. Please provide dates in DD-MM-YYYY format.")
      sys.exit(1)

    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Reservoir Index:", arg3)
    print('--------------------------------------')

    for y in range(int(start_date.year), int(end_date.year) + 1):
      print('#############################################################')
      print("Current download year:", y)
      print('#############################################################')
      if calendar.isleap(y):
        days = 366
      else:
        days = 365
      main_hls_function(y, days, arg3)

    
  else:
    print("Error: Please provide two datetime arguments in DD-MM-YYYY format and an integer as the third argument.")
    sys.exit(1)
