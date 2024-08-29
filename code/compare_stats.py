# import all necessary libraries
import sys
import os
import traceback
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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def compare_stats(res_idx):

  fname = '800_res.geojson'
  reservoirs = preprocess_reservoir_shp(fname)
  if pd.isna(reservoirs.loc[res_idx,'grand_id']) or reservoirs.loc[res_idx,'grand_id']==-999:
    im_name = str(0) + '_' + reservoirs.loc[res_idx,'name'].replace(" ", "_")
  else:
    im_name = str(int(reservoirs.loc[res_idx,'grand_id'])) + '_' + reservoirs.loc[res_idx,'name'].replace(" ", "_")


  df_final= pd.read_csv(str(res_idx)+'_f.csv')
  df_final['datetime'] = pd.to_datetime(df_final['datetime'])
  df_final = df_final[df_final['days']>60]

  in_situ_df = pd.read_csv('in-situ_data.csv')
  if reservoirs.loc[res_idx,'name'] == 'Mead':
    in_situ = in_situ_df[in_situ_df['RES_NAME']=='LAKE MEAD']
  else:
    mask = in_situ_df['RES_NAME'].str.lower().apply(lambda x: any(lake.lower() in x for lake in [reservoirs.loc[res_idx,'name']]))
    in_situ = in_situ_df[mask]
  in_situ['date'] = pd.to_datetime(in_situ['date'])

  fig, ax1 = plt.subplots(figsize=(15,3))

  # Plot the first line on the main axis
  ax1.plot(df_final['datetime'], df_final['lake_area'], '.-')
  ax1.plot(in_situ['date'],in_situ['area_m2']/10**6)
  ax1.set_ylabel('Lake Area')
  plt.savefig('final_plots/'+str(res_idx)+'_1.png',dpi=600)

  # Merge dataframes on 'date' and 'datetime' columns
  merged_df = pd.merge(in_situ[['date','area_m2']], df_final[['datetime','lake_area']], left_on='date', right_on='datetime', how='inner')
  merged_df['in_situ_area'] = merged_df['area_m2']/10**6

  # Calculate metrics
  mae = mean_absolute_error(merged_df['in_situ_area'], merged_df['lake_area'])
  rmse = np.sqrt(mean_squared_error(merged_df['in_situ_area'], merged_df['lake_area']))
  mbd = np.mean(merged_df['lake_area'] - merged_df['in_situ_area'])
  mpe = np.mean((merged_df['lake_area'] - merged_df['in_situ_area']) / merged_df['in_situ_area']) * 100
  mape = np.mean(np.abs((merged_df['lake_area'] - merged_df['in_situ_area']) / merged_df['in_situ_area'])) * 100
  r2 = r2_score(merged_df['in_situ_area'], merged_df['lake_area'])
  relative_bias = (mbd / np.mean(merged_df['in_situ_area'])) * 100

  # Calculate Pearson Correlation Coefficient
  pearson_corr = merged_df['in_situ_area'].corr(merged_df['lake_area'])

  # Print the result
  print("Pearson Correlation Coefficient: {:.2f}".format(pearson_corr))

  # Print metrics
  print("Mean Absolute Error (MAE): {:.2f} km2".format(mae))
  print("Root Mean Squared Error (RMSE): {:.2f} km2".format(rmse))
  print("Mean Bias Deviation (MBD): {:.2f} km2".format(mbd))
  print("Relative Bias: {:.2f}%".format(relative_bias))
  print("Mean Percentage Error (MPE): {:.2f}%".format(mpe))
  print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))
  print("Coefficient of Determination (R²): {:.2f}".format(r2))

  # Create a figure with 3 subplots
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))

  # Scatter plot
  axs[0].scatter(merged_df['in_situ_area'], merged_df['lake_area'], color='blue')
  axs[0].plot([merged_df['in_situ_area'].min(), merged_df['in_situ_area'].max()],
              [merged_df['in_situ_area'].min(), merged_df['in_situ_area'].max()], 'k--')
  axs[0].set_xlabel('In Situ Area')
  axs[0].set_ylabel('Lake Area')
  axs[0].set_title('Scatter Plot')
  axs[0].set_aspect('equal', adjustable='box')

  # Residual plot
  residuals = merged_df['lake_area'] - merged_df['in_situ_area']
  axs[1].scatter(merged_df['in_situ_area'], residuals, color='red')
  axs[1].axhline(y=0, color='k', linestyle='--')
  axs[1].set_xlabel('In Situ Area')
  axs[1].set_ylabel('Residuals')
  axs[1].set_title('Residual Plot')
  axs[1].set_aspect('equal', adjustable='box')

  # Bland-Altman plot
  mean_vals = (merged_df['lake_area'] + merged_df['in_situ_area']) / 2
  diff_vals = merged_df['lake_area'] - merged_df['in_situ_area']
  axs[2].scatter(mean_vals, diff_vals, color='green')
  axs[2].axhline(y=np.mean(diff_vals), color='k', linestyle='--')
  axs[2].set_xlabel('Mean of In Situ Area and Lake Area')
  axs[2].set_ylabel('Residual')
  axs[2].set_title('Bland-Altman Plot')
  axs[2].set_aspect('equal', adjustable='box')

  # Adjust layout
  plt.tight_layout()
  plt.savefig('final_plots/'+str(res_idx)+'_2.png',dpi=600)

  #if pearson_corr>0.7:
  #  print('Good fit!! Deleting HLS data')
  #  try:
  #    # Use shutil.rmtree to delete the folder and its contents
  #    folder_to_delete = os.path.join(str(res_idx))
  #    shutil.rmtree(folder_to_delete)
  #    print(f"Successfully deleted the folder: {folder_to_delete}")
  #  except Exception as e:
  #    print(f"An error occurred while deleting the folder: {e}")
  #else:
  #  print('Not a good fit!! Check data for {}'.format(reservoirs.loc[res_idx,'name']))

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

    compare_stats(arg3)
    
  else:
    print("Error: Please provide the reservoir index as the function argument.")
    sys.exit(1)