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

from sklearn.metrics import r2_score
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm


def interpolate_smoothed_value(day, smoothed_dict, df_sm):
  # If the day exists in df, return the smoothed value
  if day in smoothed_dict:
    return smoothed_dict[day]
  
  # Check if there are days before and after the current day in df
  before_days = df_sm[df_sm['days'] < day]
  after_days = df_sm[df_sm['days'] > day]
  
  # If no days before or after, return the nearest available smoothed value
  if before_days.empty:
    return after_days.iloc[0]['smoothed']
  if after_days.empty:
    return before_days.iloc[-1]['smoothed']
  
  # If there are days before and after, perform linear interpolation
  before_day = before_days.iloc[-1]
  after_day = after_days.iloc[0]
  
  fraction = (day - before_day['days']) / (after_day['days'] - before_day['days'])
  interpolated_value = before_day['smoothed'] + fraction * (after_day['smoothed'] - before_day['smoothed'])
  
  return interpolated_value


def outlier_detection(res_idx):
  df = pd.read_csv(str(res_idx)+'.csv')
  df['datetime'] = pd.to_datetime(df['date'],format='%Y%j')
  df = df.sort_values(by='date')

  cls_df = df.copy()
  cls_df = cls_df.sort_values(by='date')

  aveA = cls_df['enh_area1'].quantile(0.5)
  minA = df['enh_area2'].quantile(0.025)
  

  if aveA >50:
    c_alpha = 10
    f_alpha = 0.05
  elif aveA > 10:
    c_alpha = 1.5
    f_alpha = 0.02
  elif aveA > 1:
    c_alpha = 2.5
    f_alpha = 0.01
  else:
    c_alpha = 5
    f_lapha = 0.01

  df_sm = cls_df[(cls_df['cloud_cover']<c_alpha)&(cls_df['enh_area2']>minA)&(cls_df['ice_cover']<20)]

  # Convert datetime to number of days since the first date
  df_sm['days'] = (df_sm['datetime'] - df_sm['datetime'].iloc[0]).dt.days

  # Define the number of initial days to fit with a spline
  initial_days = 100

  # Split the data into initial and remaining portions
  initial_data = df_sm[df_sm['days'] <= initial_days]
  remaining_data = df_sm[df_sm['days'] > initial_days]

  # Fit a spline to the initial data
  spline = UnivariateSpline(initial_data['days'], initial_data['enh_area2'], s=50)

  # Calculate the smoothed values for the initial data
  initial_data['smoothed'] = spline(initial_data['days'])

  # Fit Lowess Smoothing to the remaining data
  frac = f_alpha  # The smoothing parameter (fraction of data to use in each neighborhood)
  smoothed = sm.nonparametric.lowess(remaining_data['enh_area2'], remaining_data['days'], frac=frac)

  # Extract smoothed values for the remaining data
  remaining_data['smoothed'] = smoothed[:, 1]

  # Concatenate the initial and remaining data back together
  df_sm = pd.concat([initial_data, remaining_data])

  # Compute the difference from the previous row
  diff_prev = abs(df_sm['smoothed'].diff())

  # Compute the difference from the next row
  diff_next = abs(df_sm['smoothed'].diff(periods=-1))

  # Determine the maximum absolute value from both differences
  df_sm['sm_diff_max'] = diff_prev.where(abs(diff_prev) < abs(diff_next), diff_next)
  df_sm['sm_diff_max'].fillna(0, inplace=True)

  for idx, row in df_sm[df_sm['sm_diff_max']>df_sm['sm_diff_max'].quantile(0.99)].iterrows():
    # Shift the rows down and up by one period
    df_sm1 = df_sm.shift(1)
    df_sm2 = df_sm.shift(-1)
  
    df_sm.loc[idx,'smoothed'] = (df_sm1.loc[idx,'smoothed']+df_sm2.loc[idx,'smoothed'])/2

  # Create a dictionary from df for faster lookup
  smoothed_dict = dict(zip(df_sm['days'], df_sm['smoothed']))

  # Compute the days column for cls_df
  cls_df['days'] = (cls_df['datetime'] - df_sm['datetime'].iloc[0]).dt.days

  # Compute the smoothed values for cls_df
  cls_df['smoothed'] = cls_df['days'].apply(interpolate_smoothed_value, args=(smoothed_dict, df_sm))

  # Compute the difference from the previous row
  diff_prev = abs(cls_df['smoothed'].diff())

  # Compute the difference from the next row
  diff_next = abs(cls_df['smoothed'].diff(periods=-1))

  # Determine the maximum absolute value from both differences
  cls_df['sm_diff_max'] = diff_prev.where(abs(diff_prev) < abs(diff_next), diff_next)
  cls_df['sm_diff_max'].fillna(0, inplace=True)

  # First interpolate
  cls_df['smoothed'].interpolate(inplace=True)

  # Then forward fill for NaNs at the beginning
  cls_df['smoothed'].fillna(method='ffill', inplace=True)

  # And backward fill for NaNs at the end
  cls_df['smoothed'].fillna(method='bfill', inplace=True)

  # Compute the rolling standard deviation with a window of 7 days
  cls_df['rolling_std'] = cls_df['enh_area1'].rolling(window=7, center=True).std()

  if aveA > 100:
    beta = 0.01
  elif aveA > 10:
    beta = 0.05
  elif aveA > 1:
    beta = 0.05
  else:
    beta = 0.25

  # Define the minimum and maximum constraints based on the 'smoothed' column
  min_constraint = 0.01 * cls_df['smoothed']
  max_constraint = beta * cls_df['smoothed']

  # Constrain the 'rolling_std' values
  cls_df['rolling_std'] = np.clip(cls_df['rolling_std'], min_constraint, max_constraint)

  # If there are any NaN values in 'rolling_std', you can handle them as needed. 
  # For example, to fill NaN values with the minimum constraint:
  cls_df['rolling_std'].fillna(max_constraint, inplace=True)

  ice_area = cls_df.iloc[0,10]
  for idx, row in cls_df.iterrows():

    if row['cloud_cover']>95:
      cls_df.loc[idx,'lake_area'] = np.nan
      continue

    if row['ice_cover']>50:
      cls_df.loc[idx,'lake_area'] = ice_area
      continue
      
    if abs(row['enh_area1'] - row['smoothed'])<2*row['rolling_std']:
      cls_df.loc[idx,'lake_area'] = row['enh_area1']
      ice_area = row['enh_area1']
    elif abs(row['enh_area2'] - row['smoothed'])<2*row['rolling_std']:
      cls_df.loc[idx,'lake_area'] = row['enh_area2']
      ice_area = row['enh_area2']
    else:
      cls_df.loc[idx,'lake_area'] = np.nan

  df_final = cls_df[cls_df['lake_area'].notnull()]
  df_final.to_csv(str(res_idx)+'_f.csv')





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

    outlier_detection(arg3)
    
  else:
    print("Error: Please provide the reservoir index as the function argument.")
    sys.exit(1)