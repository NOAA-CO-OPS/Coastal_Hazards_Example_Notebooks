
import requests
import time as time
import pandas as pd
import numpy as np
import warnings
import configparser

#this is to suppress user warnings
warnings.filterwarnings("ignore")

# %% Configuration file
CONFIG_FILE = "config.cfg"

def read_config_section(config_file, section):
    params = {}
    try:
        config = configparser.ConfigParser()
        with open(config_file) as f:
            #config.readfp(f)
            config.read_file(f)
            options = config.options(section)
            for option in options:            
                try:
                    params[option] = config.get(section, option)
                    if params[option] == -1:
                        print("Could not read option: %s" % option)                    
                except:
                    print("Exception reading option %s!" % option)
                    params[option] = None
    except configparser.NoSectionError as nse:
        print("No section %s found reading %s: %s", section, config_file, nse)
    except IOError as ioe:
        print("Config file not found: %s: %s", config_file, ioe)

    return params

# Obtain directory info from config file
data_params = read_config_section(CONFIG_FILE, "dir")
work_dir = data_params['work_dir']
run_dir = data_params['run_dir']
data_dir = data_params['data_dir']
stn_list = data_params['stn_list_file']
analysis_year = int(data_params['run_year'])


#read in external data files
model_output = pd.read_csv(data_dir+'\\'+str(analysis_year)+'_HTF_Annual_Outlook_full_data.csv',dtype={'stnID': str})
model_output['Station_ID'] = model_output.stnID.str[:7]
lats_lons_df = pd.read_csv(stn_list)
lats_lons_df['NOAA_ID'] = lats_lons_df['NOAA_ID'].astype(str)
model_output = pd.merge(model_output,lats_lons_df,left_on='Station_ID',right_on='NOAA_ID')

station_list = model_output['Station_ID']

HTF_Annual_outlook_station_stats = model_output[['stnName','Station_ID','Region']]

#get previous year prediction
product = 'htf_met_year_annual_outlook'
met_year = str(int(analysis_year - 1))
server = 'https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/'

myurl = server+product+'.json?&met_year='+met_year

#https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/htf_met_year_annual_outlook.json?&units=english

    # Read JSON file
urlResponse = requests.get(myurl)
content=urlResponse.json()

    # Convert JSON encoded data into Python dictionary
mydata = content['MetYearAnnualOutlook']

past_pred = pd.DataFrame(mydata)

HTF_Annual_outlook_station_stats['prev_highConf'] = past_pred['highConf']
HTF_Annual_outlook_station_stats['prev_lowConf'] = past_pred['lowConf']
HTF_Annual_outlook_station_stats['prev_mid'] = HTF_Annual_outlook_station_stats[['prev_highConf','prev_lowConf']].mean(axis=1)

#find the historical maximum number of annual HTF days
station_list = HTF_Annual_outlook_station_stats['Station_ID']
last_year_df = pd.DataFrame()
historic_df = pd.DataFrame()

for i in station_list:
  #get previous year prediction
  product = 'htf_met_year_annual'
  server = 'https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/'

  myurl = server+product+'.json?station='+str(i)

  urlResponse = requests.get(myurl)
  content=urlResponse.json()

  mydata = content['MetYearAnnualFloodCount']

  station_annual = pd.DataFrame(mydata)
  
  last_year_obs = station_annual[station_annual['metYear']==int(met_year)]
  last_year_df=last_year_df.append(last_year_obs)

  historic_max = station_annual['minCount'].idxmax()
  historic_max_year = station_annual.iloc[historic_max]
  historic_df = historic_df.append(historic_max_year)

  last_year_df=last_year_df.reset_index(drop=True)

  historic_df=historic_df.reset_index(drop=True)

HTF_Annual_outlook_station_stats['prev_observed'] = last_year_df['minCount']
HTF_Annual_outlook_station_stats['range'] = np.nan

#check if previous obs HTF count was within outlook range
for i in range(0,len(station_list)):
  if HTF_Annual_outlook_station_stats['prev_observed'].iloc[i] < HTF_Annual_outlook_station_stats['prev_lowConf'].iloc[i]:
    HTF_Annual_outlook_station_stats['range'].iloc[i]= 'BELOW'
  elif HTF_Annual_outlook_station_stats['prev_observed'].iloc[i] > HTF_Annual_outlook_station_stats['prev_highConf'].iloc[i]:
    HTF_Annual_outlook_station_stats['range'].iloc[i] = 'ABOVE'
  else:
      HTF_Annual_outlook_station_stats['range'].iloc[i] = 'WITHIN'

model_output['mid_pred'] = (model_output['highConf']+model_output['lowConf'])/2
HTF_new_predictions = model_output[['lowConf','highConf','mid_pred']]

HTF_Annual_outlook_station_stats['current_highConf'] = HTF_new_predictions['highConf']
HTF_Annual_outlook_station_stats['current_lowConf'] = HTF_new_predictions['lowConf']
HTF_Annual_outlook_station_stats['current_mid'] = HTF_new_predictions['mid_pred']
#find historic max HTF count per station
HTF_Annual_outlook_station_stats['historic_max_HTF_days'] = historic_df['minCount']
HTF_Annual_outlook_station_stats['historic_met_year'] = historic_df['metYear']
#check if last observed broke station record
HTF_Annual_outlook_station_stats['record_break'] = np.nan

for i in range(0,len(station_list)):
  if HTF_Annual_outlook_station_stats['prev_observed'].iloc[i] < HTF_Annual_outlook_station_stats['historic_max_HTF_days'].iloc[i]:
    HTF_Annual_outlook_station_stats['record_break'].iloc[i]= 'NO'
  else:
      HTF_Annual_outlook_station_stats['record_break'].iloc[i] = 'YES'

HTF_Annual_outlook_station_stats['2000_trend_only'] = model_output['pred_2k']

trend_only = pd.DataFrame(columns=['trend_only_highConf','trend_only_lowConf','trend_only_mid'])
#find trend only pred for enso stations
for i in range(len(model_output)):
    station = model_output.loc[i,'Station_ID']
    station_df = model_output.loc[model_output['Station_ID']==station]
    
    if station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
        x=station_df['lin_pred'].iloc[0]
        x_high=x+station_df['lin_rmse'].iloc[0]
        x_low=x-station_df['lin_rmse'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
        x=station_df['quad_pred'].iloc[0]
        x_high=x+station_df['quad_rmse'].iloc[0]
        x_low=x-station_df['quad_rmse'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Linear':
        x = np.nan
        x_high = np.nan
        x_low=np.nan
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Quadratic':
        x = np.nan
        x_high = np.nan
        x_low=np.nan
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)

    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Linear ENSO Sensitivity':
        x=station_df['avg19'].iloc[0]
        x_high=x+station_df['stdev19'].iloc[0]
        x_low=x-station_df['stdev19'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Quadratic ENSO Sensitivity':
        x=station_df['avg19'].iloc[0]
        x_high=x+station_df['stdev19'].iloc[0]
        x_low=x-station_df['stdev19'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    else:
        x = np.nan
        x_high = np.nan
        x_low=np.nan
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)

HTF_Annual_outlook_station_stats['trend_only_highConf'] = trend_only['trend_only_highConf']
HTF_Annual_outlook_station_stats['trend_only_lowConf'] = trend_only['trend_only_lowConf']
HTF_Annual_outlook_station_stats['trend_only_mid'] = trend_only['trend_only_mid']

HTF_Annual_outlook_station_stats['pred_difference'] = HTF_Annual_outlook_station_stats['current_mid'] - HTF_Annual_outlook_station_stats['trend_only_mid']

HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'] = HTF_Annual_outlook_station_stats['2000_trend_only']
HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'] = HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'].where(HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'] > 0, 0.5)

HTF_Annual_outlook_station_stats['percent_increase_trend_only'] = ((HTF_Annual_outlook_station_stats['trend_only_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']) / HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']*100)

HTF_Annual_outlook_station_stats['percent_increase_chosen_methods'] = ((HTF_Annual_outlook_station_stats['current_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']) / HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']*100)

HTF_Annual_outlook_station_stats=HTF_Annual_outlook_station_stats.drop(columns=['2000_trend_only_nozeros'])

HTF_Annual_outlook_station_stats.to_csv(f'{data_dir}\\{analysis_year}_HTF_Annual_Outlook_station_stats.csv',index=False)

print('Station stats exported')
###################################################################################################################################################################
###################################################################################################################################################################

#region stats
regions = HTF_Annual_outlook_station_stats['Region'].unique()
print(regions)

HTF_Annual_outlook_region_stats = pd.DataFrame()

HTF_Annual_outlook_region_stats['Region'] = regions

additional_regions = ['US','CONUS']

additional_df = pd.DataFrame({'Region': additional_regions})

HTF_Annual_outlook_region_stats = pd.concat([HTF_Annual_outlook_region_stats, additional_df], ignore_index=True)

HTF_Annual_outlook_region_stats.index=HTF_Annual_outlook_region_stats['Region']

#Median value (over all stations within region) of last year’s observed HTF days

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['prev_observed'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

conus_df = HTF_Annual_outlook_station_stats[~HTF_Annual_outlook_station_stats['Region'].isin(['PAC', 'CAR', 'AK'])]

median_conus = conus_df['prev_observed'].median()

median_us = HTF_Annual_outlook_station_stats['prev_observed'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'prev_observed': additional_regions},index=['US','CONUS'])

df = pd.concat([median_regions, additional_df])

HTF_Annual_outlook_region_stats['median_prev_observed'] = df

#Percentage of stations ABOVE/BELOW/WITHIN predicted range

num_station_region = pd.DataFrame(HTF_Annual_outlook_station_stats['Station_ID'].groupby(HTF_Annual_outlook_station_stats['Region']).count())

above = HTF_Annual_outlook_station_stats[HTF_Annual_outlook_station_stats['range']=='ABOVE']
num_above = pd.DataFrame(above['range'].groupby(above['Region']).count())

within = HTF_Annual_outlook_station_stats[HTF_Annual_outlook_station_stats['range']=='WITHIN']
num_within = pd.DataFrame(within['range'].groupby(within['Region']).count())

below = HTF_Annual_outlook_station_stats[HTF_Annual_outlook_station_stats['range']=='BELOW']
num_below = pd.DataFrame(below['range'].groupby(below['Region']).count())

num_station_region['percent_above'] = round((num_above['range']/num_station_region['Station_ID'])*100,2)
num_station_region['percent_within'] = round((num_within['range']/num_station_region['Station_ID'])*100,2)
num_station_region['percent_below'] = round((num_below['range']/num_station_region['Station_ID'])*100,2)

conus_num_stations = len(conus_df['Station_ID'].unique())

above = conus_df[conus_df['range']=='ABOVE']
num_above = above['range'].count()

within = conus_df[conus_df['range']=='WITHIN']
num_within = within['range'].count()

below = conus_df[conus_df['range']=='BELOW']
num_below = below['range'].count()

conus_percent_above = round((num_above/conus_num_stations)*100,2)

conus_percent_within = round((num_within/conus_num_stations)*100,2)
conus_percent_below = round((num_below/conus_num_stations)*100,2)

df_con = pd.DataFrame(index=['CONUS'])
df_con['Station_ID'] = conus_num_stations
df_con['percent_above']=conus_percent_above
df_con['percent_within']=conus_percent_within
df_con['percent_below']=conus_percent_below

num_station_region=num_station_region.append(df_con)

total_num_stations = len(HTF_Annual_outlook_station_stats['Station_ID'].unique())

above = HTF_Annual_outlook_station_stats[HTF_Annual_outlook_station_stats['range']=='ABOVE']
num_above = above['range'].count()

within = HTF_Annual_outlook_station_stats[HTF_Annual_outlook_station_stats['range']=='WITHIN']
num_within = within['range'].count()

below = HTF_Annual_outlook_station_stats[HTF_Annual_outlook_station_stats['range']=='BELOW']
num_below = below['range'].count()

total_percent_above = round((num_above/total_num_stations)*100,2)

total_percent_within = round((num_within/total_num_stations)*100,2)
total_percent_below = round((num_below/total_num_stations)*100,2)

df_tot = pd.DataFrame(index=['US'])
df_tot['Station_ID'] = total_num_stations
df_tot['percent_above']=total_percent_above
df_tot['percent_within']=total_percent_within
df_tot['percent_below']=total_percent_below

num_station_region=num_station_region.append(df_tot)
num_station_region=num_station_region.fillna(0.0)

HTF_Annual_outlook_region_stats.index=HTF_Annual_outlook_region_stats['Region']

HTF_Annual_outlook_region_stats['num_station_per_region'] = num_station_region['Station_ID']
HTF_Annual_outlook_region_stats['percent_above'] = num_station_region['percent_above']
HTF_Annual_outlook_region_stats['percent_within'] = num_station_region['percent_within']
HTF_Annual_outlook_region_stats['percent_below'] = num_station_region['percent_below']

#Median values of all station’s next year’s upper and lower model (trend+ENSO) predictions

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['current_highConf'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['current_highConf'].median()

median_us = HTF_Annual_outlook_station_stats['current_highConf'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'current_highConf': additional_regions},index=['US','CONUS'])

median_regions=median_regions.append(additional_df)

HTF_Annual_outlook_region_stats['median_current_highConf'] = median_regions

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['current_lowConf'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['current_lowConf'].median()

median_us = HTF_Annual_outlook_station_stats['current_lowConf'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'current_lowConf': additional_regions},index=['US','CONUS'])

median_regions=median_regions.append(additional_df)

HTF_Annual_outlook_region_stats['median_current_lowConf'] = median_regions

#Median value of number of days increase since year 2000 trend only to next year’s trend only prediction
HTF_Annual_outlook_station_stats['days_increase_trend_only'] = HTF_Annual_outlook_station_stats['trend_only_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only']

HTF_Annual_outlook_station_stats['days_increase_chosen_method'] = HTF_Annual_outlook_station_stats['current_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only']

conus_df = HTF_Annual_outlook_station_stats[~HTF_Annual_outlook_station_stats['Region'].isin(['PAC', 'CAR', 'AK'])]

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['days_increase_trend_only'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['days_increase_trend_only'].median()

median_us = HTF_Annual_outlook_station_stats['days_increase_trend_only'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'days_increase_trend_only': additional_regions},index=['US','CONUS'])

median_regions=median_regions.append(additional_df)

HTF_Annual_outlook_region_stats['median_days_increase_trend_only'] = median_regions

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['days_increase_chosen_method'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['days_increase_chosen_method'].median()

median_us = HTF_Annual_outlook_station_stats['days_increase_chosen_method'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'days_increase_chosen_method': additional_regions},index=['US','CONUS'])

median_regions=median_regions.append(additional_df)

HTF_Annual_outlook_region_stats['median_days_increase_chosen_method'] = median_regions

#Median of the station difference between previous two columns (to see regional influence of ENSO)
HTF_Annual_outlook_region_stats['median_days_increase_diff'] = HTF_Annual_outlook_region_stats['median_days_increase_trend_only']-HTF_Annual_outlook_region_stats['median_days_increase_chosen_method']

#Which station within region had the largest number of observed HTF days last year? What is that value?
max_regions = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats.groupby('Region')['prev_observed'].idxmax()]
max_regions = max_regions[['Station_ID','Region','prev_observed']]
max_regions.index=max_regions['Region']

max_conus = conus_df.loc[conus_df['prev_observed'].idxmax()]
max_conus_df = pd.DataFrame({'Station_ID':max_conus[1]},index=['CONUS'])
max_conus_df['Region']=max_conus[2]
max_conus_df['prev_observed']=max_conus[6]

max_regions = max_regions.append(max_conus_df)

max_us = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats['prev_observed'].idxmax()]
max_us_df = pd.DataFrame({'Station_ID':max_us[1]},index=['US'])
max_us_df['Region']=max_us[2]
max_us_df['prev_observed']=max_us[6]

max_regions = max_regions.append(max_us_df)

HTF_Annual_outlook_region_stats['ID_for_prev_highest'] = max_regions['Station_ID']
HTF_Annual_outlook_region_stats['Highest_prev_observed'] = max_regions['prev_observed']

#Which station within region is predicted to have the largest number of observed HTF days next year?  What is that value?
#want mid or highConf?
max_regions = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats.groupby('Region')['current_mid'].idxmax()]
max_regions = max_regions[['Station_ID','Region','current_mid']]
max_regions.index=max_regions['Region']

max_conus = conus_df.loc[conus_df['current_mid'].idxmax()]
max_conus_df = pd.DataFrame({'Station_ID':max_conus[1]},index=['CONUS'])
max_conus_df['Region']=max_conus[2]
max_conus_df['current_mid']=max_conus[6]

max_regions = max_regions.append(max_conus_df)

max_us = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats['current_mid'].idxmax()]
max_us_df = pd.DataFrame({'Station_ID':max_us[1]},index=['US'])
max_us_df['Region']=max_us[2]
max_us_df['current_mid']=max_us[6]

max_regions = max_regions.append(max_us_df)

HTF_Annual_outlook_region_stats['ID_for_pred_highest'] = max_regions['Station_ID']
HTF_Annual_outlook_region_stats['Highest_predicted'] = max_regions['current_mid']

HTF_Annual_outlook_region_stats.to_csv(f'{data_dir}\\{analysis_year}_HTF_Annual_Outlook_region_stats.csv',index=False)
print('Regional stats exported')
print('End of script')