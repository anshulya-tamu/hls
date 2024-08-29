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
from osgeo import gdal
import re
from tqdm import tqdm
import shutil
import calendar
import math
import joblib

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

from multiprocessing.pool import ThreadPool

def preprocess_reservoir_shp(fname):
  reservoirs = gp.read_file(fname)

  reservoirs.loc[31,'new_name'] = 'FARMER_S CREEK RESERVOIR'
  reservoirs.loc[106,'new_name'] = 'LAKE O THE PINES'

  t = ['/' in s for s in reservoirs['new_name']]
  i = 0
  res_names = ['COX LAKE','EAGLE NEST LAKE_MANOR LAKE','LAKE BALLINGER_LAKE MOONEN','LAKE OLNEY_LAKE COOPER','LAKE WINTERS_NEW LAKE WINTERS','MUSTANG LAKE']
  for res_idx in [23, 28, 56, 107, 129, 141]:
    reservoirs.loc[res_idx, 'new_name'] = res_names[i]
    i = i + 1

  i = 0
  res_names = ['LAKE GONZALES','CLEAR LAKE WA','CLEAR LAKE OR']
  for res_idx in [84, 388, 389]:
    reservoirs.loc[res_idx, 'new_name'] = res_names[i]
    i = i + 1

  for i in range(432):
    string = reservoirs.loc[i, 'new_name']
    regex = re.compile('[@!#$%^&*()<>?/\|}{~]')
    # Pass the string in search
    # method of regex object.
    if(regex.search(string) == None):
      continue
    else:
      print(i,string)

  return reservoirs

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


def terrain_shadow(dem, azimuth, zenith):
  # Convert azimuth and zenith angles to radians
  azimuth_rad = math.radians(azimuth)
  zenith_rad = math.radians(zenith)

  # Compute the slope and aspect
  dx, dy = np.gradient(dem)
  slope = np.arctan(np.sqrt(dx**2 + dy**2))
  aspect = np.arctan2(-dy, dx)

  # Compute the cosine of the incident angle
  cos_theta_i = np.cos(zenith_rad) * np.cos(slope) + np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)

  # Calculate the terrain shadow mask
  shadow_mask = cos_theta_i < 0

  return shadow_mask


def main_estimate_area(res_idx):
## Get reservoir shapefile and geometry
  fname = 'WT_432.geojson'
  reservoirs = preprocess_reservoir_shp(fname)
  im_name = str(reservoirs.loc[res_idx,'FID']) + '_' + reservoirs.loc[res_idx,'new_name'].replace(" ", "_")

  clf = joblib.load('lake_rf.pkl')
  clip_paths = [os.path.join(str(res_idx),'clipped_rasters',d) for d in os.listdir(os.path.join(str(res_idx),'clipped_rasters')) if os.path.isdir(os.path.join(os.path.join(str(res_idx),'clipped_rasters'), d))]
  merge_paths = [os.path.join(str(res_idx),'merged_rasters',d) for d in os.listdir(os.path.join(str(res_idx),'clipped_rasters')) if os.path.isdir(os.path.join(os.path.join(str(res_idx),'clipped_rasters'), d))]

  columns = ['date', 'class_area', 'terrain_area', 'enh_area1', 'enh_area2', 'threshold', 'cloud_cover']

  # Create an empty DataFrame
  if os.path.exists(os.path.join(str(res_idx)+'.csv')):
    df = pd.read_csv(os.path.join(str(res_idx)+'.csv'))
  else:
    df = pd.DataFrame(columns=columns)

  for i in range(len(clip_paths)):
    clip_path = clip_paths[i]
    merge_path = merge_paths[i]

    if os.path.exists(os.path.join(str(res_idx),'classified_rasters',clip_path.split('/')[-1])):
      print('classification results exist for {}'.format(clip_path))
      print('----------------------------------------------------------')
      continue

    ## Classification
    s30_bands = ['B10', 'B07', 'B09', 'B04', 'B03', 'B11', 'B06',
          'B12','B02', 'B8A', 'B08', 'B01', 'Fmask','B05']

    if os.path.isdir(os.path.join(clip_path,'S30')):
      ras = rio.open(os.path.join(clip_path,'S30','B01_stitched_raster.tif')).read(1)
      print(ras[ras==-9999].flatten().shape[0]/ras.flatten().shape[0])
      if ras[ras==-9999].flatten().shape[0]/ras.flatten().shape[0]>0.98:
        print('mostly empty raster')
        print(clip_path)
        print('---------------------------------------------------------')
        continue
      else:
        flag = 1
        print('Starting classification')
        for b in tqdm(s30_bands):
          data = rio.open(os.path.join(clip_path,'S30',b+'_stitched_raster.tif')).read(1).astype('float')*0.0001
          if flag:
            input_array = np.array(data)
            flag = 0
          else:
            input_array = np.dstack((input_array, np.array(data)))
        print(input_array.shape)
        X_pred = pd.DataFrame(input_array.reshape((-1, input_array.shape[-1])), columns=s30_bands)

        predictions = clf.predict(X_pred)
        print('Classification done!!')
        print('---------------------------------------------------------')

        final_img = np.array(predictions.reshape((input_array.shape[0], input_array.shape[1]))).astype('int')
    else:
      continue

    class_area =  final_img[final_img==1].shape[0]*900/10**6
    print('Current raw area from classified image is {} km2'.format(final_img[final_img==1].shape[0]*900/10**6))

    ## Terrain Correction
    ##------------------------------------

    az_data = np.array(rio.open(os.path.join(clip_path,'S30','SAA_stitched_raster.tif')).read(1).astype('float'))
    az_data[az_data==55537]=np.nan
    azimuth = np.nanmean(az_data)*0.01

    zen_data = np.array(rio.open(os.path.join(clip_path,'S30','SZA_stitched_raster.tif')).read(1).astype('float'))
    zen_data[zen_data==55537]=np.nan
    zenith = np.nanmean(zen_data)*0.01

    print('Solar azimuth angle: {}\tSolar zenith angle {}'.format(azimuth, zenith))

    dem_ras = rio.open(os.path.join('srtm_data',im_name+'.tif'))
    resample_jrc_raster(os.path.join(clip_path,'S30', 'B01_stitched_raster.tif'), dem_ras, os.path.join(clip_path,'S30','dem.tif'))
    dem_ras = rio.open(os.path.join(clip_path,'S30','dem.tif'))
    # show(dem_ras)
    dem = dem_ras.read(1)

    print('Shape of the classified image is {} and DEM is {}'.format(final_img.shape, dem.shape))

    # Compute the terrain shadow mask
    shadow_mask = terrain_shadow(dem, azimuth, zenith)
    sm = 1 - shadow_mask.astype(int)

    # Input the JRC data
    jrc = rio.open(os.path.join(clip_path,'S30','jrc.tif'))
    jrc_data = jrc.read(1)
    print('Shape of the JRC image is {}'.format(jrc_data.shape))

    or_jrc_data = rio.open(os.path.join('jrc_data',im_name+'.tif')).read(1)

    X = final_img.astype(float)
    X[(sm==0)&(X==1)] = 2
    X[(jrc_data==0)] = 0
    terr_img = X

    # fig, ax = plt.subplots(1,2,figsize=(12,6))
    # ax[0].imshow(final_img)
    # ax[0].set_xlabel('Raw Classified Image')
    # ax[1].imshow(X)
    # ax[1].set_xlabel('Image after terrain correction')
    # plt.show()

    terr_area = X[X==1].shape[0]*900/10**6
    print('Current raw area after terrain correction is {} km2'.format(X[X==1].shape[0]*900/10**6))

    ## Image Enhancement
    ##------------------------------------------

    meta_df = pd.read_csv(os.path.join(clip_path, 'tile_metadata.csv'))
    cl_cov = np.mean(meta_df['cloud_cover'])

    print('cloud cover: {}'.format(cl_cov))

    water_mask = (X == 1).astype(int)
    # plt.imshow(water_mask)
    # plt.xlabel('Water mask after terrain correction')
    # plt.show()

    Y = water_mask*jrc_data
    hist1, _ = np.histogram(Y.flatten(),bins=np.linspace(1,100,100))
    hist2, _ = np.histogram(jrc_data.flatten(),bins=np.linspace(1,100,100))

    ratio = []
    for i in range(len(hist1)):
      ratio_value = hist1[i] / hist2[i] if hist2[i] != 0 else 0  # Handle division by zero
      ratio.append(ratio_value)

    # plt.plot(hist1/hist2)
    # plt.axhline(y=0.15,color='r')
    # plt.title('Relative proportion of JRC pixels')
    # plt.show()

    w_mask = water_mask[water_mask==1].shape[0]/jrc_data[jrc_data>0].shape[0]
    print('water mask proportion {}'.format(w_mask))

    y = 0.15

    # Estimate the threshold
    f_x = np.array(ratio)
    x_values = np.linspace(1,100,100)
    tolerance = 0.05  # Define a suitable tolerance level

    indices = np.where(np.isclose(f_x, y, atol=tolerance))

    x_for_y = x_values[indices]
    print(x_for_y)

    if len(x_for_y)>0:
      thr = np.median(x_for_y)
    else:
      thr = np.nan
    print('threshold: {}'.format(thr))

    if cl_cov < 20:
      X[(X==2)&(jrc_data>=thr)] = 1
      X[(jrc_data<thr)] = 0
      np.save('correction_coefficient.npy', f_x)
    else:
      X[jrc_data>=thr] = 1
      X[jrc_data<thr] = 0

    enh1_area = X[X==1].shape[0]*900/10**6
    print('Area after terrain correction and enhancement is {} km2'.format(X[X==1].shape[0]*900/10**6))

    if cl_cov>20 and X.shape[0]>1000:
      if os.path.exists('correction_coefficient.npy'):
        enh2_area = 0
        f_x = np.load('correction_coefficient.npy')
        for enhp in range(1,100):
          if enhp<thr:
            continue
          enh2_area = enh2_area + or_jrc_data[(or_jrc_data>=enhp)&(or_jrc_data<enhp+1)].shape[0]*900/10**6*f_x[enhp-1]
      else:
        enh2_area = or_jrc_data[or_jrc_data>thr].shape[0]*900/10**6
    else:
      enh2_area = enh1_area

    print('Area after terrain correction and enhancement is {} km2'.format(enh2_area))

    enhanced_mask = (or_jrc_data>thr).astype(int)

    # fig, ax = plt.subplots(1,2,figsize=(12,6))
    # ax[0].imshow(X)
    # ax[0].set_xlabel('final enhanced classified image')
    # ax[1].imshow(enhanced_mask)
    # ax[1].set_xlabel('Original JRC enhanced image')
    # plt.show()

    new_row = pd.DataFrame([{'date': merge_path.split('/')[-1],
               'class_area': class_area,
               'terrain_area': terr_area,
               'enh_area1': enh1_area,
               'enh_area2': enh2_area,
               'threshold': thr,
               'cloud_cover': cl_cov}])

    df = pd.concat([df, new_row], ignore_index=True)

    print('Added to the dataframe!!!')
    print('-----------------------------------------------------------')

    df.to_csv(str(res_idx)+'.csv', index=False)

    dir_path = os.path.join(str(res_idx),'classified_rasters',clip_path.split('/')[-1])
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f"Directory {dir_path} created.")
    else:
      print(f"Directory {dir_path} already exists.")

    np.save(os.path.join(dir_path,'raw.npy'), final_img)
    np.save(os.path.join(dir_path,'terrain.npy'), terr_img)
    np.save(os.path.join(dir_path,'jrc.npy'), jrc_data)
    np.save(os.path.join(dir_path,'enhanced1.npy'), X)
    np.save(os.path.join(dir_path,'enhanced2.npy'), enhanced_mask)





if __name__ == "__main__":
  # Check if arguments are provided
  if len(sys.argv) >= 2:

    try:
      arg3 = int(sys.argv[1])  # Convert the third argument to an integer
    except ValueError:
      print("Error: The reservoir index argument must be an integer.")
      sys.exit(1)

    print("Reservoir Index:", arg3)
    print('--------------------------------------')

    main_estimate_area(arg3)
    
  else:
    print("Error: Please provide the reservoir index as the function argument.")
    sys.exit(1)