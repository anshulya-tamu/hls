import os
import argparse
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import geopandas as gp

def get_ice_periods(df_):
    zero_periods = []
    start_date = None
    
    for i in range(len(df_)):
        if df_['Below_Zero'].iloc[i]:
            # If start_date is None, this is the start of a new cold period
            if start_date is None:
                start_date = df_['Date'].iloc[i]
        elif start_date is not None:
            # End of the cold period when temperature goes above 0°C
            end_date = df_['Date'].iloc[i - 1]
            # Record the period using the minimum temperature as the start
            zero_periods.append({'Start': start_date, 'End': end_date})
            # Reset for the next potential cold period
            start_date = None
            min_temp = None
            min_temp_date = None
    
    # Handle the case where the last period in the data is a below-zero period
    if start_date is not None:
        zero_periods.append({'Start': start_date, 'End': df_['Date'].iloc[-1]})
    
    # Convert to DataFrame for easier viewing
    zero_periods_df = pd.DataFrame(zero_periods)
    return zero_periods_df

# Define a function to filter and compute 'lake_area'
def calculate_lake_area_new(row, stdArea):
    values_in_range = []

    # if row['flag']=='O':
    #     return row['raw_area'], 'O'

    # if row['flag'] == 'ICE' and row['cloud_cover']<0.05:
    #     return row['raw_area'], 'ICE'

    # Check if each value is within the smoothed_area ± stdArea range
    if (row['raw_area'] >= row['smoothed_area'] - stdArea) and (row['raw_area'] <= row['smoothed_area'] + stdArea):
        values_in_range.append(row['raw_area'])

    if (row['enh1_area'] >= row['smoothed_area'] - stdArea) and (row['enh1_area'] <= row['smoothed_area'] + stdArea):
        values_in_range.append(row['enh1_area'])

    if (row['enh2_area'] >= row['smoothed_area'] - stdArea) and (row['enh2_area'] <= row['smoothed_area'] + stdArea):
        values_in_range.append(row['enh2_area'])

    # If there are values in range, return their average
    if values_in_range:
        return np.mean(values_in_range), 'E'
    else:
        return row['smoothed_area'], 'I'  # Indicate no valid value

def process_reservoir(gid, base_dir):
    try:
    
        grand = gp.read_file(os.path.join(base_dir, 'data/auxiliary/gis/hls_reservoirs.geojson'))
        file = os.path.join(base_dir, f'results/csv-files/{gid}.csv')
    
        cls_df = pd.read_csv(file)
        cls_df['datetime'] = pd.to_datetime(cls_df['date'], format = '%Y%j')
        
        print(gid, grand.loc[grand['GRAND_ID']==gid, 'RES_NAME'].values[0], grand.loc[grand['GRAND_ID']==gid, 'DAM_NAME'].values[0])
        
        # gadf = pd.read_table(f'/scratch/user/anshulya/hls/data/gang_data/{gid}_intp')
        # gadf['datetime'] = pd.to_datetime(gadf['1month'])
        # gadf = gadf[(gadf['datetime']>datetime(2015,1,1))&(gadf['datetime']<datetime(2017,1,1))]
        
        if 'enh1_area' not in cls_df.keys():
            print('No enhanced area found')
            bad_res.append(gid)
            return
        else:
            for col in ['raw_area', 'enh1_area', 'enh2_area']:
                cls_df[col] = cls_df[col]
        
        if np.nanmax(cls_df['cloud_cover'])>1:
            cls_df['cloud_cover'] = cls_df['cloud_cover']/100
    
        if np.nanmax(cls_df['ice_cover'])>1:
            cls_df['ice_cover'] = cls_df['ice_cover']/100
    
        cls_df['ice_cover'] = cls_df['ice_cover'].fillna(0)
        cls_df['eff_cloud_cover'] = cls_df['cloud_cover'] + cls_df['ice_cover']
        cls_df['eff_cloud_cover'] = cls_df['eff_cloud_cover'].clip(upper=1)
        
        cls_df = cls_df[(cls_df['enh1_area']>0)]
        cls_df = cls_df.sort_values(by='datetime')
        cls_df['year'] = [x.year for x in cls_df['datetime']]
        cls_df['month'] = [x.month for x in cls_df['datetime']]
    
        minA = grand.loc[grand['GRAND_ID']==gid, 'AREA_SKM'].values[0]*0.1
    
        #############################################################################
        ########################### ICE COVER PERCENTAGE ############################
        #############################################################################
        
        ice_df = pd.read_csv(os.path.join(base_dir, 'data/auxiliary/ice/ice_temperature_grand.csv'))

        # Start and end dates
        start_date = datetime(2016, 1, 1)
        end_date = datetime(2023, 12, 1)
        
        # Generate list of dates (first day of each month)
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            # Move to the first day of the next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        ice_df_ = pd.DataFrame({'Date': date_list, 'Temperature':ice_df[ice_df['GID']==gid].values[0][1:]})
        ice_df_['Date'] = pd.to_datetime(ice_df_['Date'])  # Ensure 'Date' is in datetime format
        ice_df_['Below_Zero'] = ice_df_['Temperature'] <= 0.25
        zero_periods_df = get_ice_periods(ice_df_)
        if zero_periods_df.empty:
            cls_df['ice_cover'] = 0
        else:
            for idx, row in zero_periods_df.iterrows():
                st_date = row['Start']
                if not st_date.year == 2016:
                    if st_date.month == 12:
                        st_date = datetime(st_date.year + 1, 1, 1)
                    else:
                        st_date = datetime(st_date.year, st_date.month + 1, 1)
                    
                end_date = row['End']
                if end_date.month == 12:
                    end_date = datetime(end_date.year + 1, 2, 1)
                else:
                    end_date = datetime(end_date.year, end_date.month + 2, 1)
                cls_df.loc[(cls_df['datetime']>st_date)&(cls_df['datetime']<end_date), 'ice_cover'] = cls_df.loc[(cls_df['datetime']>st_date)&(cls_df['datetime']<end_date), 'ice_cover'] + cls_df.loc[(cls_df['datetime']>st_date)&(cls_df['datetime']<end_date), 'cloud_cover'] 
            
            for idx, row in ice_df_[ice_df_['Below_Zero']==False].iterrows():
                month = row['Date'].month
                year = row['Date'].year
                cls_df.loc[(cls_df['year']==year)&(cls_df['month']==month), 'ice_cover'] = 0
                
        cls_df['ice_cover'] = cls_df['ice_cover'].clip(upper=1)
        
        # fig, ax = plt.subplots(2,1,figsize=(10,3), sharex = True)
        # ax[0].plot(cls_df['datetime'],cls_df['raw_area'],'r.')
        # ax[0].plot(cls_df['datetime'],cls_df['enh1_area'],'k.', alpha=0.5)
        # # ax[0].plot(gadf['datetime'], gadf['3water_enh'])
    
        # ax[1].plot(cls_df['datetime'],cls_df['cloud_cover'],color='gray')
        # ax[1].plot(cls_df['datetime'],cls_df['ice_cover'],color='blue')
        # plt.show()
    
        #############################################################################
        ########################### MAKE LOWESS SMOOTHING ###########################
        #############################################################################
    
        caveA = grand.loc[grand['GRAND_ID']==gid, 'AREA_SKM'].values[0]
        if caveA > 100:
            lowess_cloud =  0.05
        elif caveA > 20:
            lowess_cloud =  0.1
        elif caveA > 5:
            lowess_cloud =  0.25
        else:
            lowess_cloud =  0.35
        
    
        df_filtered_raw = cls_df[(cls_df['eff_cloud_cover'] < 0.25)].copy()
        df_filtered_raw['lowess_area'] = df_filtered_raw.apply(lambda row: row['raw_area'] if row['cloud_cover'] < 0.1 else row['enh1_area'], axis=1)
    
        df_cloudless = df_filtered_raw[df_filtered_raw['cloud_cover'] < 0.01]
        
        # plt.figure(figsize=(10,1.75))
        # plt.plot(df_filtered_raw['datetime'],df_filtered_raw['lowess_area'],'g.', alpha=0.5)
        # plt.plot(df_cloudless['datetime'],df_cloudless['lowess_area'],'r*', alpha=0.5)
        # plt.show()
    
        # if gid in drop_res.keys():
        #     df_filtered_raw = df_filtered_raw[df_filtered_raw['lowess_area'] > drop_res[int(gid)]]
    
        df_combined = df_filtered_raw.copy()
        df_combined['lowess_area'] = df_combined['lowess_area'].fillna(df_combined['enh1_area'])
        df_combined = df_combined[df_combined['lowess_area'] > minA]
        df_combined = df_combined.sort_values(by='datetime')
        
        x = pd.to_datetime(df_combined['datetime']).map(pd.Timestamp.toordinal).values  # Convert dates to ordinal values
        y = df_combined['lowess_area'].fillna(method='ffill').values  # Fill missing raw areas for continuity
        
        # Perform weighted LOWESS smoothing
        lowess_smoothed = sm.nonparametric.lowess(y, x, frac=0.035)
        
        # Extract smoothed values
        df_combined['smoothed_area'] = lowess_smoothed[:, 1]
        
        df_combined.loc[df_combined['datetime']<datetime(2015,6,1),'smoothed_area'] = df_combined.loc[df_combined['datetime']<datetime(2015,6,1),'lowess_area']
        
        # plt.figure(figsize=(10,2))
        # plt.plot(df_combined['datetime'], df_combined['lowess_area'], 'r.', label='Raw Area', alpha=0.5)
        # # plt.plot(df_combined['datetime'], df_combined['enh2_area'], 'b.', label='Enhanced Area', alpha=0.5)
        # plt.plot(df_combined['datetime'], df_combined['smoothed_area'], 'k', label='Smoothed Area')
        # plt.legend()
        # plt.show()
    
        #############################################################################
        ########################### LOWESS SPIKE DETECTION ##########################
        #############################################################################
        
        window = 10 # days
        threshold = 4  # example slope-difference threshold
        outlier_idx = []
        
        df_combined["is_downward_spike"] = False
        
        for idx, row in df_combined.iterrows():
            if idx < window:
                continue
            
            z_df = df_combined[idx - window : idx + window]
            # print(left_start, right_end, z_df.shape[0])
        
            if z_df.shape[0]<5:
                continue
        
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(z_df['smoothed_area'].dropna()))
        
            # Define a threshold for Z-score, typically 3 is a good cutoff for outliers
            if len(z_df.loc[z_scores > threshold].index.values)>0:
                for iidex in z_df.loc[z_scores > threshold].index.values:
                    df_combined.loc[iidex, "is_downward_spike"] = True
                    outlier_idx.append(iidex)
        print(outlier_idx)            
        df_combined = df_combined[~(df_combined["is_downward_spike"])]
        # Interpolate the NaN values
        df_combined['smoothed_area'] = df_combined['smoothed_area'].interpolate()
        
        
        # plt.figure(figsize=(10,2))
        # plt.plot(df_combined['datetime'], df_combined['raw_area'], 'r.', label='Raw Area')
        # plt.plot(df_combined['datetime'], df_combined['smoothed_area'], 'k.-', label='Smoothed Area')
        # plt.plot(df_combined.loc[df_combined['is_downward_spike'],'datetime'], df_combined.loc[df_combined['is_downward_spike'],'smoothed_area'], 'b*', label='Spikes')
        # plt.legend()
    
        #############################################################################
        ########################### MAKE FINAL DATASET ##############################
        #############################################################################
    
        aveA = np.nanmax(df_combined['smoothed_area'])
        if aveA > 100:
            beta = 0.01
        elif aveA > 20:
            beta = 0.025
        elif aveA > 5:
            beta = 0.05
        else:
            beta = 0.1
        
        flag = 0
        for i in range(2):
            if not flag == 1:
                flag = 1
            else:
                if val<0.75:
                    if aveA > 100:
                        beta = beta + 0.01
                    elif aveA > 20:
                        beta = beta + 0.025
                    elif aveA > 5:
                        beta = beta + 0.025
                    else:
                        beta = beta + 0.05
                else:
                    continue
            
            stdArea = aveA*beta
            print('Outlier Detection starts here:\n---------------------------------------------')
            
            # plt.figure(figsize=(10,2))
            
            # # Plot the original data
            # plt.plot(cls_df['datetime'], cls_df['raw_area'], '.', alpha=0.25)
            # plt.plot(cls_df['datetime'], cls_df['enh1_area'], '.', alpha=0.25)
            # plt.plot(cls_df['datetime'], cls_df['enh2_area'], '.', alpha=0.25)
            
            # # Plot the smoothed area
            # plt.plot(df_combined['datetime'], df_combined['smoothed_area'], 'k')
            
            # # Plot the shaded region around the smoothed area
            # plt.fill_between(
            #     df_combined['datetime'],
            #     df_combined['smoothed_area'] - stdArea,   # Lower bound
            #     df_combined['smoothed_area'] + stdArea,   # Upper bound
            #     color='red', alpha=0.7                   # Customize color and transparency
            # )
            
            # plt.show()
            
            merged_df = cls_df.merge(df_combined[['datetime', 'smoothed_area']], on='datetime', how='left')
    
            print('##TEST###: ', merged_df.shape, cls_df.shape)
    
            # Set 'datetime' as the index to interpolate based on it
            merged_df.set_index('datetime', inplace=True)
            merged_df['smoothed_area'] = merged_df['smoothed_area'].interpolate(method='time')
            merged_df.reset_index(inplace=True)
    
            #-------------------------- MAKE FLAG VALUES --------------------------
            merged_df['flag'] = 'I'
            
            for idx, row in zero_periods_df.iterrows():
                merged_df.loc[(merged_df['datetime']>row['Start'])&(merged_df['datetime']<row['End'])&(merged_df['ice_cover']>0.15),'flag'] = 'P-ICE'
                merged_df.loc[(merged_df['datetime']>row['Start'])&(merged_df['datetime']<row['End'])&(merged_df['ice_cover']>0.5),'flag'] = 'ICE'
    
            for idx, row in merged_df.iterrows():
                if row['cloud_cover'] < 0.01 and row['flag'] != 'ICE':
                    merged_df.loc[idx, 'flag'] = 'O'
            
            # Apply the function to each row
            merged_df['lake_area'], merged_df['flag'] = zip(*merged_df.apply(lambda row: calculate_lake_area_new(row, stdArea), axis=1))
            merged_df = merged_df.dropna(subset=['lake_area'])
    
            x = pd.to_datetime(merged_df['datetime']).map(pd.Timestamp.toordinal).values  # Convert dates to ordinal values
            y = merged_df['lake_area'].fillna(method='ffill').values  # Fill missing raw areas for continuity
            
            # Perform weighted LOWESS smoothing
            lowess_smoothed = sm.nonparametric.lowess(y, x, frac=0.025)
            merged_df['smoothed_area'] = lowess_smoothed[:, 1]
    
            merged_df.sort_values(by='datetime', inplace=True)
            
            print('Final surface area curve:\n----------------------------------------')
            # fig, ax = plt.subplots(2,1,figsize=(10,5),sharex=True)
            # ax[0].plot(merged_df['datetime'], merged_df['lake_area'], 'r.', alpha=0.5)
            # ax[0].plot(merged_df['datetime'], merged_df['smoothed_area'], 'k', linewidth=2)
    
            # ax[1].plot(merged_df['datetime'], merged_df['cloud_cover'], color='gray', alpha = 0.2)
            # ax[1].plot(merged_df['datetime'], merged_df['ice_cover'], color='blue', alpha = 0.5)
            # plt.show()
    
            for idx, row in zero_periods_df.iterrows():
                merged_df.loc[(merged_df['datetime']>row['Start'])&(merged_df['datetime']<row['End'])&(merged_df['ice_cover']>0.15),'flag'] = 'P-ICE'
                merged_df.loc[(merged_df['datetime']>row['Start'])&(merged_df['datetime']<row['End'])&(merged_df['ice_cover']>0.5),'flag'] = 'ICE'
            
            val = merged_df[merged_df['flag'] == 'I'].shape[0]/merged_df[(merged_df['flag'] != 'ICE')&(merged_df['flag'] != 'P-ICE')].shape[0]
            print(val)
    
            if val<0.25:
                break
        
        for idx, row in merged_df.iterrows():
            if row['cloud_cover'] < 0.01 and row['flag'] != 'ICE':
                merged_df.loc[idx, 'flag'] = 'O'
                
        # plt.figure(figsize=(10,2))
        # plt.plot(merged_df.loc[merged_df['flag']=='O','datetime'], merged_df.loc[merged_df['flag']=='O','lake_area'], 'b*')
        # plt.plot(merged_df.loc[merged_df['flag']=='E','datetime'], merged_df.loc[merged_df['flag']=='E','lake_area'], 'r.')
        # plt.plot(merged_df.loc[merged_df['flag']=='I','datetime'], merged_df.loc[merged_df['flag']=='I','lake_area'], 'm.', alpha=0.25)
        # for d in merged_df.loc[merged_df['flag']=='ICE','datetime']:
        #     plt.axvline(d, color='#89CFF0', alpha=0.5)
        # for d in merged_df.loc[merged_df['flag']=='P-ICE','datetime']:
        #     plt.axvline(d, color='#ADD8E6', alpha=0.5)
        # plt.plot(merged_df['datetime'], merged_df['smoothed_area'], linewidth = 2, color = 'k')
        # plt.show()
    
        merged_df = merged_df[['datetime', 'raw_area', 'cloud_cover', 'ice_cover', 'lake_area', 'smoothed_area', 'flag']]
        merged_df.to_csv(os.path.join(base_dir, f'results/final_files/{gid}.csv'), index=False)
        # return merged_df
    except Exception as e:
        tb_str = traceback.format_exc()
        print("An error occurred:")
        print(tb_str)
        print('------------------ERROR--------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process reservoir data for a given GRAND ID")
    parser.add_argument('--gid', type=int, required=True, help='GRAND reservoir ID')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing input data')

    args = parser.parse_args()
    process_reservoir(args.gid, args.base_dir)