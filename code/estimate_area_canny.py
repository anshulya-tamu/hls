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
import cv2

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

def find_elbow(data):
  """
  Find the elbow point in a dataset.
  
  Parameters:
  - data: A list or numpy array of y-values.
  
  Returns:
  - index of the elbow point.
  """
  # Create a vector from the first data point to the last data point
  first_point = [0, data[0]]
  last_point = [len(data) - 1, data[-1]]
  line_vector = np.subtract(last_point, first_point)
  
  # Normalize the line vector
  line_vector_norm = line_vector / np.sqrt(sum(line_vector**2))
  
  # Find the point that is farthest from the line
  max_distance = -1
  elbow_index = 0
  for i, value in enumerate(data):
    point = [i, value]
    distance = np.abs(np.cross(line_vector_norm, np.subtract(first_point, point)))
    if distance > max_distance:
      max_distance = distance
      elbow_index = i
            
  return elbow_index

def half_image_correction(larger_raster_path, smaller_raster_path, padded_raster_path):
  """
  Pads the smaller raster to match the size of the larger raster.
  
  Parameters:
  - larger_raster_path: Path to the larger raster file.
  - smaller_raster_path: Path to the smaller raster file.
  - padded_raster_path: Path to save the padded raster.
  """
  
  with rio.open(larger_raster_path) as larger_raster:
    with rio.open(smaller_raster_path) as smaller_raster:
        
      # Extract pixel resolution from the raster's transform
      pixel_resolution_x = abs(smaller_raster.transform[0])
      pixel_resolution_y = abs(smaller_raster.transform[4])

      # Calculate the difference in extents
      left_diff = int((smaller_raster.bounds.left - larger_raster.bounds.left) / pixel_resolution_x)
      right_diff = int((larger_raster.bounds.right - smaller_raster.bounds.right) / pixel_resolution_x)
      top_diff = int((larger_raster.bounds.top - smaller_raster.bounds.top) / pixel_resolution_y)
      bottom_diff = int((smaller_raster.bounds.bottom - larger_raster.bounds.bottom) / pixel_resolution_y)
      
      # Read the smaller raster data
      smaller_data = smaller_raster.read(1)
      
      # Pad the smaller raster data
      padded_data = np.pad(smaller_data, ((top_diff, bottom_diff), (left_diff, right_diff)), 'constant', constant_values=-9999)
      
      # Create a new raster with the padded data and the same metadata as the larger raster
      with rio.open(padded_raster_path, 'w', **larger_raster.meta) as dst:
        dst.write(padded_data, 1)

def process_clip_path(clip_path, merge_path, clf, res_idx, im_name, sat, bands):
  try:
    print('Processing: ',clip_path)

    # Correct for partial images
    extent_df = pd.read_csv('max_img_extent.csv')
    extent_df = extent_df[extent_df['res_idx']==res_idx]


    if sat == 'S30':
      max_x = extent_df.iloc[0,4]
      max_y = extent_df.iloc[0,5]
      max_file_name = os.path.join(extent_df.iloc[0,7],'S30/B05_stitched_raster.tif')
    else:
      max_x = extent_df.iloc[0,2]
      max_y = extent_df.iloc[0,3]
      max_file_name = os.path.join(extent_df.iloc[0,6],'L30/B05_stitched_raster.tif')


    for b in tqdm(bands):
      ras = rio.open(os.path.join(clip_path,sat,b+'_stitched_raster.tif')).read(1)
      if ras.shape[0]<max_x or ras.shape[1]<max_y:
        half_image_correction(max_file_name,
            os.path.join(clip_path,sat,b+'_stitched_raster.tif'),
            os.path.join(clip_path,sat,b+'_stitched_raster.tif'))
      else:
        print('Full image band: ',b)

    
    # Check if image has substantial data
    ras = rio.open(os.path.join(clip_path,sat,'B01_stitched_raster.tif')).read(1)
    print(ras[ras==-9999].flatten().shape[0]/ras.flatten().shape[0])
    if ras[ras==-9999].flatten().shape[0]/ras.flatten().shape[0]>0.98:
      print('mostly empty raster')
      print(clip_path)
      print('---------------------------------------------------------')
      return
    else:
      flag = 1
      print('Starting classification')
      for b in tqdm(bands):
        data = rio.open(os.path.join(clip_path,sat,b+'_stitched_raster.tif')).read(1).astype('float')*0.0001
        if flag:
          input_array = np.array(data)
          flag = 0
        else:
          input_array = np.dstack((input_array, np.array(data)))
      print(input_array.shape)
    
    test_df = pd.DataFrame(input_array.reshape((-1, input_array.shape[-1])), columns=bands)
    if sat == 'S30':
      denominator = test_df['B03'] + test_df['B08']
      test_df['ndwi'] = np.where(denominator != 0, (test_df['B03'] - test_df['B08']) / denominator, 0)
      # Set values beyond the range of -1 to 1 as np.nan
      test_df['ndwi'] = np.where((test_df['ndwi'] > 1) | (test_df['ndwi'] < -1), -1, test_df['ndwi'])
    elif sat == 'L30':
      #L30 NDWI
      denominator = test_df['B03'] + test_df['B05']
      test_df['ndwi'] = np.where(denominator != 0, (test_df['B03'] - test_df['B05']) / denominator, 0)
      #L30 MNDWI calculation
      denominator = test_df['B03'] + test_df['B07']
      test_df['mndwi'] = np.where(denominator != 0, (test_df['B03'] - test_df['B07']) / denominator, 0)
  
      # Set values beyond the range of -1 to 1 as np.nan
      test_df['ndwi'] = np.where((test_df['ndwi'] > 1) | (test_df['ndwi'] < -1), -1,test_df['ndwi'])
      # Set values beyond the range of -1 to 1 as np.nan
      test_df['mndwi'] = np.where((test_df['mndwi'] > 1) | (test_df['mndwi'] < -1), -1, test_df['mndwi'])
        
    X_pred = test_df

    t1 = datetime.now()

    predictions = clf.predict(X_pred)
    print('prediction took {} time'.format(datetime.now()-t1))
    print('Classification done!!')
    print('---------------------------------------------------------')

    final_img = np.array(predictions.reshape(input_array.shape[:2])).astype('int')

    ## Terrain Correction
    ##------------------------------------

    az_data = np.array(rio.open(os.path.join(clip_path,sat,'SAA_stitched_raster.tif')).read(1).astype('float'))
    az_data[az_data==55537]=np.nan
    azimuth = np.nanmean(az_data)*0.01

    zen_data = np.array(rio.open(os.path.join(clip_path,sat,'SZA_stitched_raster.tif')).read(1).astype('float'))
    zen_data[zen_data==55537]=np.nan
    zenith = np.nanmean(zen_data)*0.01

    print('Solar azimuth angle: {}\tSolar zenith angle {}'.format(azimuth, zenith))

    dem_ras = rio.open(os.path.join('srtm_data',im_name+'.tif'))
    resample_jrc_raster(os.path.join(clip_path,sat, 'B01_stitched_raster.tif'), dem_ras, os.path.join(clip_path,sat,'dem.tif'))
    dem_ras = rio.open(os.path.join(clip_path,sat,'dem.tif'))
    dem = dem_ras.read(1)

    print('Shape of the classified image is {} and DEM is {}'.format(final_img.shape, dem.shape))

    # Compute the terrain shadow mask
    shadow_mask = terrain_shadow(dem, azimuth, zenith)
    sm = 1 - shadow_mask.astype(int)

    # Input the JRC data
    jrc_ras = rio.open(os.path.join('jrc_data',im_name+'.tif'))
    resample_jrc_raster(os.path.join(clip_path,sat, 'B01_stitched_raster.tif'), jrc_ras, os.path.join(clip_path,sat,'jrc.tif'))
    jrc_ras = rio.open(os.path.join(clip_path,sat,'jrc.tif'))
    jrc_data = jrc_ras.read(1)
    print('Shape of the JRC image is {}'.format(jrc_data.shape))

    ## Calculate the cloud cover
    ref_img = rio.open(os.path.join(clip_path,sat, 'B01_stitched_raster.tif')).read(1).astype('float')*0.0001

    X = final_img.astype(float)
    X[(ref_img==-9999*0.0001)] = 2
    X[(sm==0)&(X==1)] = 2
    X[(jrc_data==0)] = 0
    terr_img = X.copy()


    b = ref_img[ref_img!=-9999*0.0001].shape[0]
    a = X[X==2].shape[0]
    cl_cov = a/b*100
    print('Cloud cover in % over reservoir: {:.2f}'.format(cl_cov))

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

    if cl_cov < 1:
        print('Cloud cover less than 1 %. No enhancement required!!')
        bin_value = 0
        enh_area1 = terr_area
        enh_area2 = terr_area

        dir_path = os.path.join(str(res_idx),'classified_rasters',clip_path.split('/')[-1],sat)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory {dir_path} created.")
        else:
            print(f"Directory {dir_path} already exists.")

        # np.save(os.path.join(dir_path,'raw.npy'), final_img)
        np.save(os.path.join(dir_path,'terrain.npy'), terr_img)
        np.save(os.path.join(dir_path,'enhanced1.npy'), terr_img)
        np.save(os.path.join(dir_path,'enhanced2.npy'), terr_img)

        new_row = pd.DataFrame([{'date': merge_path.split('/')[-1],
                   'terrain_area': terr_area,
                   'enh_area1': enh_area1,
                   'enh_area2': enh_area2,
                   'threshold': np.nan,
                   'elbow': np.nan,
                   'cloud_cover': cl_cov}])

        # df = pd.concat([df, new_row], ignore_index=True)

        print('Added to the dataframe!!!')
        print('----------------------------------------------------------------------------------')
    else:
        a = 0
        b = 30
        Y = np.zeros(X.shape)
        Y[X==1] = b
        Y[X==0] = a
        Y[X==2] = 100
        # Create a mask for vertical transitions between a and b
        vertical_mask = np.where(((Y == a) & (np.roll(Y, shift=-1, axis=1) == b)) |
                              ((Y == b) & (np.roll(Y, shift=-1, axis=1) == a)), 255, 0).astype(np.uint8)

        # Create a mask for horizontal transitions between a and b
        horizontal_mask = np.where(((Y == a) & (np.roll(Y, shift=-1, axis=0) == b)) |
                                ((Y == b) & (np.roll(Y, shift=-1, axis=0) == a)), 255, 0).astype(np.uint8)

        # Combine the masks
        combined_mask = np.maximum(vertical_mask, horizontal_mask)

        # Apply Canny edge detection
        edges = cv2.Canny(combined_mask.astype(np.uint8), 20, 40)

        # Compute the histogram
        hist, bins = np.histogram(jrc_data[(edges>0)&(X!=0)].flatten(), bins=100, density=True)

        # Compute the CDF
        cdf = np.cumsum(hist) * (bins[1] - bins[0])  # Multiply by bin width to ensure the CDF goes to 1

        # Plot the histogram and CDF
        # plt.figure(figsize=(7, 3))

        # plt.subplot(1, 2, 1)
        # plt.bar(bins[:-1], hist, width=bins[1] - bins[0])
        # plt.title('Histogram (PDF)')

        # plt.subplot(1, 2, 2)
        # plt.plot(bins[:-1], cdf, marker='o')
        # plt.title('CDF')
        # plt.axhline(y=0.8,color='red')
        # plt.axvline(x=find_elbow(cdf),color='green')
        # plt.ylim(0, 1)

        # plt.tight_layout()
        # plt.show()

        # Find the bin value corresponding to the first CDF value just greater than 0.8
        index = np.where(cdf >= 0.8)[0][0]
        bin_value = bins[index]
        thr = int(bin_value)

        print(f"The bin value corresponding to the first CDF value just greater than 0.8 is: {bin_value}")
        
        Y = X.copy()
        Y[(jrc_data>=int(bin_value))] = 1
        print('Area after enhancement from thresholding is {} km2'.format(Y[Y==1].shape[0]*900/10**6))
        enh_area1 = Y[Y==1].shape[0]*900/10**6

        # plt.imshow(Y)
        # plt.xlabel('Thresholding enhancement')
        # plt.show()
        
        print('Elbow index: ',find_elbow(cdf))

        elbow_point = find_elbow(cdf)
    
        # Compute the rate of change
        rate_of_change = np.diff(cdf)
        
        # Calculate average rate of change before and after the elbow point
        avg_rate_before = np.mean(rate_of_change[:elbow_point])
        avg_rate_after = np.mean(rate_of_change[elbow_point:])

        # Generate the "hockeyness" index
        hockeyness_index = avg_rate_after - avg_rate_before

        print(f"Hockeyness Index: {hockeyness_index}")
        
        # if hockeyness_index >0.0:
        bin_value = find_elbow(cdf)

        X[(jrc_data>=int(bin_value))] = 1
        enh_area2 = X[X==1].shape[0]*900/10**6
        print('Area after elbow enhancement is {} km2'.format(X[X==1].shape[0]*900/10**6))

        # plt.imshow(X)
        # plt.xlabel('Elbow enhancement')
        # plt.show()

        dir_path = os.path.join(str(res_idx),'classified_rasters',clip_path.split('/')[-1],sat)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory {dir_path} created.")
        else:
            print(f"Directory {dir_path} already exists.")

        # np.save(os.path.join(dir_path,'raw.npy'), final_img)
        np.save(os.path.join(dir_path,'terrain.npy'), terr_img)
        np.save(os.path.join(dir_path,'enhanced1.npy'), Y)
        np.save(os.path.join(dir_path,'enhanced2.npy'), X)

        new_row = pd.DataFrame([{'date': merge_path.split('/')[-1],
                   'terrain_area': terr_area,
                   'enh_area1': enh_area1,
                   'enh_area2': enh_area2,
                   'threshold': thr,
                   'elbow': int(bin_value),
                   'cloud_cover': cl_cov}])

        # df = pd.concat([df, new_row], ignore_index=True)

        print('Added to the dataframe!!!')
        print('----------------------------------------------------------------------------------')

    return new_row
    
  except BaseException as error:
    print('An exception occurred: {}'.format(error))
    print('----------------------------------------------------------------------------------')
    return



def main_estimate_area(res_idx):
## Get reservoir shapefile and geometry
  fname = 'WT_432.geojson'
  reservoirs = preprocess_reservoir_shp(fname)
  im_name = str(reservoirs.loc[res_idx,'FID']) + '_' + reservoirs.loc[res_idx,'new_name'].replace(" ", "_")

  clip_paths = [os.path.join(str(res_idx),'clipped_rasters',d) for d in os.listdir(os.path.join(str(res_idx),'clipped_rasters')) if os.path.isdir(os.path.join(os.path.join(str(res_idx),'clipped_rasters'), d))]
  merge_paths = [os.path.join(str(res_idx),'merged_rasters',d) for d in os.listdir(os.path.join(str(res_idx),'clipped_rasters')) if os.path.isdir(os.path.join(os.path.join(str(res_idx),'clipped_rasters'), d))]

  columns = ['date', 'terrain_area', 'enh_area1', 'enh_area2', 'threshold', 'elbow', 'cloud_cover']

  # Create an empty DataFrame
  if os.path.exists(os.path.join(str(res_idx)+'.csv')):
    df = pd.read_csv(os.path.join(str(res_idx)+'.csv'))
  else:
    df = pd.DataFrame(columns=columns)

  for i in range(len(clip_paths)):
    clip_path = clip_paths[i]
    merge_path = merge_paths[i]

    if os.path.isdir(os.path.join(clip_path,'S30')):
      if os.path.exists(os.path.join(str(res_idx),'classified_rasters',clip_path.split('/')[-1],'S30')):
        print('classification results exist for S30 -> {}'.format(clip_path))
        print('----------------------------------------------------------')
      else:
        s30_bands = ['B01', 'B02', 'B03', 'B04', 'B05','B06','B07','B08','B8A','B09','B10', 'B11', 'B12']
        clf = joblib.load('lake_rf_new.pkl')
        sat = 'S30'
        new_row = process_clip_path(clip_path, merge_path, clf, res_idx, im_name, sat, s30_bands)

        df = pd.concat([df, new_row],ignore_index=True)
        df.to_csv(os.path.join(str(res_idx)+'.csv'),index=False)
    
    if os.path.isdir(os.path.join(clip_path,'L30')):
      if os.path.exists(os.path.join(str(res_idx),'classified_rasters',clip_path.split('/')[-1],'L30')):
        print('classification results exist for L30 -> {}'.format(clip_path))
        print('----------------------------------------------------------')
      else:
        l30_bands = ['B01', 'B02', 'B03', 'B04', 'B05','B06','B07','B10','B11']
        clf = joblib.load('lake_rf_new_L30.pkl')
        sat = 'L30'
        new_row = process_clip_path(clip_path, merge_path, clf, res_idx, im_name, sat, l30_bands)
        
        df = pd.concat([df, new_row],ignore_index=True)
        df.to_csv(os.path.join(str(res_idx)+'.csv'),index=False)


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