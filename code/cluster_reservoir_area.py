# import all necessary libraries
import os
import sys
from datetime import datetime, timedelta
import requests as r
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import random
# from osgeo import gdal
from pathlib import Path
import io
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz

import zipfile
from collections import defaultdict

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

import joblib
import math
import cv2
from scipy import sparse
import traceback

import pyproj
from pyproj import Proj

from shapely.ops import transform
from shapely.geometry import Polygon

import multiprocessing
from multiprocessing.pool import ThreadPool
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import time
import random


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
            
            
## Merge downloaded rasters
def merge_max_rasters(res_gid, bands, l30_rasters_to_merge, save_path, sat):
    if not os.path.exists(os.path.join(save_path, sat)):
        os.makedirs(os.path.join(save_path, sat))
    
    for b in tqdm(bands):
        bandpaths = []
        for dirpath in l30_rasters_to_merge:
            bandpaths.append([os.path.join(dirpath, bname) for bname in os.listdir(dirpath) if b in bname][0])
        # print(bandpaths)
        rasters = [rio.open(raster_path) for raster_path in bandpaths]
    
        # Merge the rasters into a single dataset
        b1_merged, b1_merged_transform = merge(rasters)
    
        # Update the metadata of the merged dataset
        b1_merged_meta = rasters[0].meta.copy()
        b1_merged_meta.update({
            'height': b1_merged.shape[1],
            'width': b1_merged.shape[2],
            'transform': b1_merged_transform
        })
        
        if not os.path.exists(os.path.join(save_path, sat)):
            os.makedirs(os.path.join(save_path, sat))
        
        # Create a new raster file to store the stitched data
        output_path = os.path.join(save_path, sat, b +'_stitched_raster.tif')
        # print(output_path)
        with rio.open(output_path, 'w', **b1_merged_meta) as dst:
            dst.write(b1_merged)
        
        # Close all the input rasters
        for raster in rasters:
            raster.close()

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

## Merge downloaded rasters
def merge_rasters(dirpaths, bands, save_path, sat):
    
    for b in tqdm(bands):
        bandpaths = []
        for dirpath in dirpaths:
            bandpaths.append([os.path.join(dirpath, bname) for bname in os.listdir(dirpath) if b in bname][0])
        # print(bandpaths)
        rasters = [rio.open(raster_path) for raster_path in bandpaths]
    
        # Merge the rasters into a single dataset
        b1_merged, b1_merged_transform = merge(rasters)
    
        # Update the metadata of the merged dataset
        b1_merged_meta = rasters[0].meta.copy()
        b1_merged_meta.update({
            'height': b1_merged.shape[1],
            'width': b1_merged.shape[2],
            'transform': b1_merged_transform
        })
        
        if not os.path.exists(os.path.join(save_path, sat)):
            os.makedirs(os.path.join(save_path, sat))
        
        # Create a new raster file to store the stitched data
        output_path = os.path.join(save_path, sat, b +'_stitched_raster.tif')
        # print(output_path)
        with rio.open(output_path, 'w', **b1_merged_meta) as dst:
            dst.write(b1_merged)
        
        # Close all the input rasters
        for raster in rasters:
            raster.close()

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

def get_bit(number, bit_position):
  return (number >> bit_position) & 1

def matrix_get_bit(matrix, bit_position):
  matrix = matrix.astype(int)  # Ensure matrix elements are integers
  vec_get_bit = np.vectorize(get_bit)
  return vec_get_bit(matrix, bit_position)
  
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
      
      
def classify_reservoir(res_gid, clip_path, sat):

    try:
        if sat == 'L30':
            bands = ['B01', 'B02', 'B03', 'B04', 'B05','B06','B07','B10','B11']
            clf = joblib.load(os.path.join('/scratch/user/anshulya/hls/data/models/rf_models', 'lake_rf_new_L30_ice.pkl'))
        else:
            bands = ['B01', 'B02', 'B03', 'B04', 'B05','B06','B07','B08','B8A','B09','B10', 'B11', 'B12']
            clf = joblib.load(os.path.join('/scratch/user/anshulya/hls/data/models/rf_models', 'lake_rf_new_S30_ice.pkl'))
        
        print('Processing: ',clip_path)
        
        if sat == 'S30':
            max_file_name = f'/scratch/user/anshulya/hls/data/max_rasters/{res_gid}/S30/B01_clipped_raster.tif'
            max_x, max_y = rio.open(max_file_name).read(1).shape
        else:
            max_file_name = f'/scratch/user/anshulya/hls/data/max_rasters/{res_gid}/L30/B01_clipped_raster.tif'
            max_x, max_y = rio.open(max_file_name).read(1).shape
        
        ext_bands = bands + ['Fmask']  
        for b in ext_bands:
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
            return None, None, None
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
        
        super_df = pd.DataFrame(input_array.reshape((-1, input_array.shape[-1])), columns=bands)
        test_df = super_df[(super_df != -0.9999).all(axis=1)]
        
        if sat == 'S30':
            # NDSI calculation for Sentinel-2 (Green and SWIR1)
            denominator_ndsi = test_df['B03'] + test_df['B11']
            test_df['ndsi'] = np.where(denominator_ndsi != 0, (test_df['B03'] - test_df['B11']) / denominator_ndsi, 0)
            # Set values beyond the range of -1 to 1 as -10
            test_df['ndsi'] = np.where((test_df['ndsi'] > 1) | (test_df['ndsi'] < -1), -10, test_df['ndsi'])
        
            # NDVI (Red and NIR)
            denominator_ndvi = test_df['B08'] + test_df['B04']
            test_df['ndvi'] = np.where(denominator_ndvi != 0, (test_df['B08'] - test_df['B04']) / denominator_ndvi, 0)
            # Set values beyond the range of -1 to 1 as -10
            test_df['ndvi'] = np.where((test_df['ndvi'] > 1) | (test_df['ndvi'] < -1), -10, test_df['ndvi'])
        
            # NDMI (NIR and SWIR1)
            denominator_ndmi = test_df['B08'] + test_df['B11']
            test_df['ndmi'] = np.where(denominator_ndmi != 0, (test_df['B08'] - test_df['B11']) / denominator_ndmi, 0)
            # Set values beyond the range of -1 to 1 as -10
            test_df['ndmi'] = np.where((test_df['ndmi'] > 1) | (test_df['ndmi'] < -1), -10, test_df['ndmi'])
        
            # NDWI (Green and NIR)
            denominator = test_df['B03'] + test_df['B08']
            test_df['ndwi'] = np.where(denominator != 0, (test_df['B03'] - test_df['B08']) / denominator, 0)
        
            # MNDWI calculation (Green and SWIR2)
            denominator = test_df['B03'] + test_df['B12']
            test_df['mndwi'] = np.where(denominator != 0, (test_df['B03'] - test_df['B12']) / denominator, 0)
            # Set values beyond the range of -1 to 1 as -10
            test_df['mndwi'] = np.where((test_df['mndwi'] > 1) | (test_df['mndwi'] < -1), -10, test_df['mndwi'])
        
            # AWEI without shadow
            test_df['awei_ns'] = 4 * (test_df['B03'] - test_df['B11']) - (0.25 * test_df['B08'] + 2.75 * test_df['B12'])
        
            # AWEI with shadow (assuming 'Blue' corresponds to Band 2)
            test_df['awei_sh'] = test_df['B02'] + 2.5 * test_df['B03'] - 1.5 * (test_df['B08'] + test_df['B11']) - 0.25 * test_df['B12']
        
        elif sat == 'L30':
            
            ## NDSI calculation for Landsat (Green and SWIR1)
            denominator_ndsi = test_df['B03'] + test_df['B06']
            test_df['ndsi'] = np.where(denominator_ndsi != 0, (test_df['B03'] - test_df['B06']) / denominator_ndsi, 0)
            # Set values beyond the range of -1 to 1 as np.nan
            test_df['ndsi'] = np.where((test_df['ndsi'] > 1) | (test_df['ndsi'] < -1), -1, test_df['ndsi'])
        
            ## Calculation of NDVI (Red and NIR) for comparison
            denominator_ndvi = test_df['B05'] + test_df['B04']
            test_df['ndvi'] = np.where(denominator_ndvi != 0, (test_df['B05'] - test_df['B04']) / denominator_ndvi, 0)
            # Set values beyond the range of -1 to 1 as np.nan
            test_df['ndvi'] = np.where((test_df['ndvi'] > 1) | (test_df['ndvi'] < -1), -1, test_df['ndvi'])
        
            ## Calculation of other indices (e.g., Normalized Difference Moisture Index - NDMI)
            denominator_ndmi = test_df['B05'] + test_df['B06']
            test_df['ndmi'] = np.where(denominator_ndmi != 0, (test_df['B05'] - test_df['B06']) / denominator_ndmi, 0)
            # Set values beyond the range of -1 to 1 as np.nan
            test_df['ndmi'] = np.where((test_df['ndmi'] > 1) | (test_df['ndmi'] < -1), -1, test_df['ndmi'])
        
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
        
            # AWEI without shadow
            test_df['awei_ns'] = 4 * (test_df['B03'] - test_df['B06']) - (0.25 * test_df['B05'] + 2.75 * test_df['B07'])
        
            # AWEI with shadow (assuming 'Blue' corresponds to Band 2)
            test_df['awei_sh'] = test_df['B02'] + 2.5 * test_df['B03'] - 1.5 * (test_df['B05'] + test_df['B06']) - 0.25 * test_df['B07']
        X_pred = test_df
        
        t1 = datetime.now()
        
        predictions = clf.predict(X_pred)
        print('prediction took {} time'.format(datetime.now()-t1))
        print('Classification done!!')
        print('---------------------------------------------------------')
        
        valid_rows_mask = (super_df != -0.9999).all(axis=1)
        super_df['predictions'] = 0
        super_df.loc[valid_rows_mask, 'predictions'] = predictions
        predictions = super_df['predictions'].values
        final_img = np.array(predictions.reshape(input_array.shape[:2])).astype('int')
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(final_img)
        
        ## Terrain Correction
        ##------------------------------------
        
        az_data = np.array(rio.open(os.path.join(clip_path,sat,'SAA_stitched_raster.tif')).read(1).astype('float'))
        az_data[az_data>=55537]=np.nan
        azimuth = np.nanmean(az_data)*0.01
        
        zen_data = np.array(rio.open(os.path.join(clip_path,sat,'SZA_stitched_raster.tif')).read(1).astype('float'))
        zen_data[zen_data>=55537]=np.nan
        zenith = np.nanmean(zen_data)*0.01
        
        print('Solar azimuth angle: {}\tSolar zenith angle {}'.format(azimuth, zenith))
        
        im_name = [f for f in os.listdir(os.path.join('/scratch/user/anshulya/hls/data/auxiliary/srtm_data')) if f.startswith(str(res_gid)+'_')][0]
        
        dem_ras = rio.open(os.path.join('/scratch/user/anshulya/hls/data/auxiliary/srtm_data',im_name))
        resample_jrc_raster(os.path.join(clip_path,sat,'B01_stitched_raster.tif'), dem_ras, os.path.join(clip_path,sat,'dem.tif'))
        dem_ras = rio.open(os.path.join(clip_path,sat,'dem.tif'))
        dem = dem_ras.read(1)
        
        print('Shape of the classified image is {} and DEM is {}'.format(final_img.shape, dem.shape))
        
        # Compute the terrain shadow mask
        shadow_mask = terrain_shadow(dem, azimuth, zenith)
        sm = 1 - shadow_mask.astype(int)
        
        im_name = [f for f in os.listdir(os.path.join('/scratch/user/anshulya/hls/data/auxiliary/jrc_data')) if f.startswith(str(res_gid)+'_')][0]
        
        # Input the JRC data
        jrc_ras = rio.open(os.path.join('/scratch/user/anshulya/hls/data/auxiliary/jrc_data',im_name))
        resample_jrc_raster(os.path.join(clip_path,sat, 'B01_stitched_raster.tif'), jrc_ras, os.path.join(clip_path,sat,'jrc.tif'))
        jrc_ras = rio.open(os.path.join(clip_path,sat,'jrc.tif'))
        jrc_data = jrc_ras.read(1)
        print('Shape of the JRC image is {}'.format(jrc_data.shape))
        
        ## Calculate the cloud cover
        ref_img = rio.open(os.path.join(clip_path,sat,'B01_stitched_raster.tif')).read(1).astype('float')*0.0001
        
        X = final_img.astype(float)
        X[(ref_img==-9999*0.0001)] = 2
        
        fmask = rio.open(os.path.join(clip_path,sat,'Fmask_stitched_raster.tif')).read(1).astype('int')
        cloud_mask = matrix_get_bit(fmask, 1)
        cloud_shadow = matrix_get_bit(fmask, 3)
        water_mask = matrix_get_bit(fmask, 5)
        ice_mask = matrix_get_bit(fmask, 4)
        
        raw_cloud = X.copy()
        raw_cloud[(water_mask==1)] = 1
        raw_cloud[(cloud_mask==1)&(X!=1)] = 2
        raw_cloud[(cloud_shadow==1)] = 2
        raw_cloud[(ice_mask==1)|(X==3)] = 3
        raw_cloud[(ref_img==-9999*0.0001)] = -1
        
        i_frac = ice_mask[(ice_mask==1)&(X==3)&(jrc_data>0)].shape[0]/jrc_data[jrc_data>0].shape[0]
        print('Ice fraction is: {:.2f}%'.format(i_frac*100))
        
        if X[X==1].shape[0]==0:
            print('No water found! Probably very cloudy!!')
            return final_img, final_img, final_img
        else:
            w_frac = (raw_cloud[raw_cloud==1]).shape[0]/(X[X==1]).shape[0]
            print('Water ratio (Fmask:our classification) is: {:.2f}'.format(w_frac))
            
            X[(sm==0)&(X==1)] = 2
            X[(jrc_data==0)] = 0
        
            terr_img = X.copy()
            terr_area = X[X==1].shape[0]*900/10**6
            print('Current raw area after terrain correction is {} km2'.format(X[X==1].shape[0]*900/10**6))
        
            b = jrc_data[jrc_data>0].shape[0]
            a = X[(X==2)&(jrc_data>0)].shape[0]
            cl_cov = a/b*100
            print('Cloud cover in % over reservoir: {:.2f}'.format(cl_cov))
        
            # fig, ax = plt.subplots(1,2,figsize=(6,3))
            # ax[0].imshow(final_img)
            # ax[0].set_xlabel('Raw Classified Image')
            # ax[1].imshow(X)
            # ax[1].set_xlabel('Image after terrain correction')
            # plt.show()
        
            if i_frac>0.9 and cl_cov<10:
                print('Ice fraction is more than 90%! Area estimation not performed')
                return terr_img, terr_img, terr_img
                
            if cl_cov < 5:
                print('Cloud cover less than 5 %. No enhancement required!!')
                bin_value = 0
                enh_area1 = terr_area
                enh_area2 = terr_area
                return terr_img, terr_img, terr_img
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
                
                # # Plot the histogram and CDF
                # plt.figure(figsize=(6, 3))
                
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
                if i_frac>0.1:
                    Y[(jrc_data>=int(bin_value))&(X<=2)]=1
                else:
                    Y[(jrc_data>=int(bin_value))]=1
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
                
                if i_frac>0.1:
                    X[(jrc_data>=int(bin_value))&(X<=2)] = 1
                else:
                    X[(jrc_data>=int(bin_value))] = 1
                
                enh_area2 = X[X==1].shape[0]*900/10**6
                print('Area after elbow enhancement is {} km2'.format(X[X==1].shape[0]*900/10**6))
                
                # plt.imshow(X)
                # plt.xlabel('Elbow enhancement')
                # plt.show()
                return terr_img, Y, X
    except KeyboardInterrupt:
        print('Process interrupted by user.')
        return None, None, None
    except BaseException as error:
        print('An exception occurred: {}'.format(error))
        traceback.print_exc()
        print('----------------------------------------------------------------------------------')
        return None, None, None
        
        
def process_reservoirs_for_day(day, todo_tiles, todo_res_list, grand):
    day_tiles = []

    # Check which tiles exist for the given day
    for tile in todo_tiles:
        if os.path.exists(f'/scratch/user/anshulya/hls/data/cluster/{tile}/unmerged_rasters/{day}'):
            day_tiles.append(tile)

    day_res = [res for res in todo_res_list.keys() if any(tile in day_tiles for tile in todo_res_list[res])]

    # Loop over reservoirs
    for res_gid in day_res:
        print('Processing starting for reservoir: ', res_gid, day)
        
        result_directory_path = os.path.join('/scratch/user/anshulya/hls/results/hls_classified', str(res_gid), str(day))
        if os.path.isdir(result_directory_path):
            print('Directory already exists')
            print('##########################################')
            continue
            
        merge_path = f'/scratch/user/anshulya/hls/data/raw/{res_gid}/merged_rasters/{day}'
        clip_path = f'/scratch/user/anshulya/hls/data/raw/{res_gid}/clipped_rasters/{day}'

        res = grand.loc[grand['GRAND_ID'] == res_gid]
        res.geometry = res.geometry.buffer(0.01)

        tiles = [t for t in todo_res_list[res_gid] if t in day_tiles]

        # Process for each satellite (L30, S30)
        for sat in ['L30', 'S30']:
            all_exist = all(Path(f'/scratch/user/anshulya/hls/data/cluster/{tile}/unmerged_rasters/{day}/{sat}').exists() for tile in tiles)
            if not all_exist:
                continue
            
            print('Projecting raster tiles')
            for tile in tiles:
                dirpath = f'/scratch/user/anshulya/hls/data/cluster/{tile}/unmerged_rasters/{day}/{sat}'
                reproject_unmerged_rasters(dirpath)

            if sat == 'L30':
                bands = ['B01', 'B03', 'B05', 'B06', 'B04', 'Fmask', 'B07',
                         'B10', 'B02', 'B11', 'SAA', 'SZA']
            else:
                bands = ['B10', 'B07', 'B09', 'B04', 'B03', 'B11', 'B06',
                         'B12', 'B02', 'B8A', 'B08', 'B01', 'Fmask', 'B05', 'SAA', 'SZA']

            dirpaths = [f'/scratch/user/anshulya/hls/data/cluster/{tile}/unmerged_rasters/{day}/{sat}' for tile in tiles]
            merge_rasters(dirpaths, bands, merge_path, sat)

            if not os.path.exists(os.path.join(clip_path, sat)):
                os.makedirs(os.path.join(clip_path, sat))

            # Clip rasters
            for f in tqdm(os.listdir(f'{merge_path}/{sat}/')):
                ras = rio.open(f'{merge_path}/{sat}/{f}')
                cr_name = f.split('_')[0] + '_stitched_raster.tif'
                if not os.path.exists(f'{clip_path}/{sat}/{cr_name}'):
                    clip_raster(ras, res, f'{clip_path}/{sat}/{cr_name}')

            # Classify reservoir
            A, B, C = classify_reservoir(res_gid, clip_path, sat)
            if A is not None:
                results_dir = os.path.join('/scratch/user/anshulya/hls/results/hls_classified', str(res_gid), str(day))
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                # Save the results as sparse matrices
                sparse.save_npz(os.path.join(results_dir, 'terrain.npz'), sparse.csr_matrix(A))
                sparse.save_npz(os.path.join(results_dir, 'enh1.npz'), sparse.csr_matrix(B))
                sparse.save_npz(os.path.join(results_dir, 'enh2.npz'), sparse.csr_matrix(C))

        # Clean up the directories after processing
        shutil.rmtree(merge_path)
        shutil.rmtree(clip_path)

def process_dates_in_range(start_day, end_day, num_workers):
    grand = gp.read_file('/scratch/user/anshulya/hls/data/auxiliary/gis/hls_reservoirs.geojson')
    old_hls = gp.read_file('/scratch/user/anshulya/hls/data/auxiliary/gis/800_res.geojson')
    s2_res = pd.read_csv('/scratch/user/anshulya/hls/data/auxiliary/gis/sentinel_tiles.csv')
    
    
    not_done_gid = []
    old_done_idx = [int(f.split('_')[0]) for f in os.listdir('/scratch/user/anshulya/hls/results/old') if f.endswith('_f.csv')]
    old_gid = [old_hls.loc[id,'grand_id'] for id in old_done_idx]
    
    new_gid = [int(f) for f in os.listdir('/scratch/user/anshulya/hls/results/hls_classified') if int(f) not in old_gid]
    files = ['2016.zip','2017.zip','2018.zip','2019.zip','2020.zip','2021.zip','2022.zip','2023.zip']
    for g in tqdm(new_gid):
        all_exist = all(Path(f'/scratch/user/anshulya/hls/results/hls_classified/{g}/{f}').exists() for f in files)
        if not all_exist:
            not_done_gid.append(g)
    new_gid = [int(f) for f in os.listdir('/scratch/user/anshulya/hls/results/hls_classified') if int(f) not in old_gid and int(f) not in not_done_gid]
    print(len(os.listdir('/scratch/user/anshulya/hls/results/hls_classified')), len(new_gid))
    
    done_gid = new_gid + old_gid
    not_done_gid = [x for x in grand['GRAND_ID'].values if x not in done_gid]
    
    todo_tiles = ['15SXT', '16SDF', '16SCE', '16SDC', '16SEC', '16SFA', '16SCG', '16SDB', '17SMS', '16SDG', '16RFV', '16SCF', '16SDE', '16SED', '16RGV', '16SGA']
                  
    # res_list = [x for x in s2_res.loc[s2_res['Name'].isin(todo_tiles),'GRAND_ID'].unique() if x in not_done_gid]
    
    # todo_res_list = {}
    # ext_res = []
    # for r in res_list:
    #     tiles = s2_res.loc[s2_res['GRAND_ID']==r,'Name'].unique()
    #     rogue = [t for t in tiles if t not in todo_tiles]
    #     if len(rogue)>0:
    #         print(r, tiles)
    #         print(r, [t for t in tiles if t not in todo_tiles])
    #         ext_res.append(r)
    #     else:
    #         todo_res_list[r] = list(tiles)
    # print('Reservoirs to be processed: ',len(todo_res_list))
    
    todo_res_list = {1132: ['15SXT'],
                     1142: ['15SXT'],
                     1910: ['16RFV', '16SFA'],
                     1912: ['16RFV'],
                     1916: ['16RFV', '16RGV'],
                     1753: ['16SCE', '16SCF', '16SCG', '16SDE', '16SDF'],
                     1752: ['16SCF', '16SCG', '16SDF', '16SDG'],
                     1887: ['16SDB'],
                     1888: ['16SDB', '16SDC'],
                     1867: ['16SDC', '16SEC'],
                     1874: ['16SDC'],
                     1792: ['16SDE'],
                     1749: ['16SDF', '16SDG'],
                     1847: ['16SEC', '16SED'],
                     1909: ['16SGA'],
                     1891: ['17SMS'],
                     1893: ['17SMS']}
    
    ###############
    # todo_res_list[1493] = ['16TCP']

    # Use ProcessPoolExecutor to parallelize the function
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to process each day in parallel
        futures = [executor.submit(process_reservoirs_for_day, day, todo_tiles, todo_res_list, grand) for day in range(start_day, end_day)]
    
        # Optionally, wait for all tasks to complete and handle results
        for future in futures:
            try:
                future.result()  # This will raise any exceptions encountered during execution
            except Exception as e:
                print(f"Error processing day: {e}")
                
                
if __name__ == '__main__':
    # Check if the required arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python cluster_reservoir_area.py <start_date> <end_date> <num_workers>")
        print("Example: python script.py 01-01-2022 12-31-2022 4")
        sys.exit(1)
    
    # Get the command-line arguments
    start_date_input = sys.argv[1]
    end_date_input = sys.argv[2]
    num_workers_input = int(sys.argv[3])

    start_date = datetime.strptime(start_date_input, '%m-%d-%Y')
    end_date = datetime.strptime(end_date_input, '%m-%d-%Y')
    
    print('Processing: ', int(start_date.strftime('%Y%j')), ' to ', int(end_date.strftime('%Y%j')))

    process_dates_in_range(int(start_date.strftime('%Y%j')), int(end_date.strftime('%Y%j')), int(num_workers_input))
  




