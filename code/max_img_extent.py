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
import concurrent.futures


def generate_max_img_extent(res_idx):
  S30_x = []
  S30_y = []
  L30_x = []
  L30_y = []
    
  columns = ['file', 'sat', 's30_x', 's30_y', 'l30_x', 'l30_y']
  df = pd.DataFrame(columns=columns)

    
  folders = [os.path.join(str(res_idx),'clipped_rasters',d) for d in os.listdir(os.path.join(str(res_idx),'clipped_rasters'))]
  for f in tqdm(folders):
    sat = [s for s in os.listdir(f) if os.path.isdir(os.path.join(f,s))]
    try:
      for s in sat:
        if s=='S30':
          ras_data = rio.open(os.path.join(f,s,'B04_stitched_raster.tif')).read(1).astype('float')
          new_row = pd.DataFrame({
                        'file': f,
                        'sat': s,
                        's30_x': ras_data.shape[0],
                        's30_y': ras_data.shape[1],
                        'l30_x': np.nan,
                        'l30_y': np.nan,
                    }, index=[0])
          df = pd.concat([df,new_row],ignore_index=True)
        if s=='L30':
          ras_data = rio.open(os.path.join(f,s,'B04_stitched_raster.tif')).read(1).astype('float')
          new_row = pd.DataFrame({
                        'file': f,
                        'sat': s,
                        'l30_x': ras_data.shape[0],
                        'l30_y': ras_data.shape[1],
                        's30_x': np.nan,
                        's30_y': np.nan,
                    }, index=[0])
          df = pd.concat([df,new_row],ignore_index=True)
    except BaseException as error:
      print('An exception occurred: {}'.format(error))
    
    # ## Sentinel raster sizes
    # fig = plt.figure(figsize=(15, 10))
    # k = 0
    # for x in df['s30_x'].unique():
    #     for y in df['s30_y'].unique():
    #         df_ = df[(df['s30_x']==x)&(df['s30_y']==y)]
    #         if not df_.empty:
    #             ax = fig.add_subplot(6, 5, k + 1)
    #             ras = rio.open(os.path.join(df_.iloc[0,0],df_.iloc[0,1],'B04_stitched_raster.tif'))
    #             show(ras, ax=ax)
    #             ax.set_title('{}, {}'.format(x,y))
    #             ax.set_xlabel(df_.iloc[0,0])
    #             k = k + 1
    # plt.tight_layout()
    # plt.show()

    # ## Landsat raster sizes
    # fig = plt.figure(figsize=(15, 10))
    # k = 0
    # for x in df['l30_x'].unique():
    #     for y in df['l30_y'].unique():
    #         df_ = df[(df['l30_x']==x)&(df['l30_y']==y)]
    #         if not df_.empty:
    #             ax = fig.add_subplot(5, 5, k + 1)
    #             ras = rio.open(os.path.join(df_.iloc[0,0],df_.iloc[0,1],'B04_stitched_raster.tif'))
    #             show(ras, ax=ax)
    #             ax.set_title('{}, {}'.format(x,y))
    #             ax.set_xlabel(df_.iloc[0,0])
    #             k = k + 1
    # plt.tight_layout()
    # plt.show()

  df['pixel_area_l30'] = df['l30_x']*df['l30_y']
  df['pixel_area_s30'] = df['s30_x']*df['s30_y']

  df['pixel_area_l30'] = df['pixel_area_l30'].astype('float')
  df['pixel_area_s30'] = df['pixel_area_s30'].astype('float')

  if os.path.exists('max_'+str(res_idx)+'.csv'):
    # Load the CSV file into the DataFrame
    extent_df = pd.read_csv('max_'+str(res_idx)+'.csv')
  else:
    # Create an empty DataFrame with the specified columns
    columns = ['res_idx', 'L30_x', 'L30_y', 'S30_x', 'S30_y', 'l_file', 's_file']
    extent_df = pd.DataFrame(columns=columns)
    
  new_row = pd.DataFrame({
       'res_idx': res_idx, 
        'L30_x': df.loc[df['pixel_area_l30'].idxmax()]['l30_x'], 
        'L30_y': df.loc[df['pixel_area_l30'].idxmax()]['l30_y'], 
        'S30_x': df.loc[df['pixel_area_s30'].idxmax()]['s30_x'], 
        'S30_y': df.loc[df['pixel_area_s30'].idxmax()]['s30_y'], 
        'l_file': df.loc[df['pixel_area_l30'].idxmax()]['file'], 
        's_file': df.loc[df['pixel_area_s30'].idxmax()]['file']
  },index=[0])
    
  extent_df = pd.concat([extent_df, new_row], ignore_index=True)

  extent_df.to_csv('max_'+str(res_idx)+'.csv', index=False)
  return




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

    generate_max_img_extent(arg3)
    
  else:
    print("Error: Please provide the reservoir index as the function argument.")
    sys.exit(1)