import requests
import pandas as pd
#from plotnine import *
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap # pip install basemap
import ast
import numpy as np
import os
import configparser

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

pd.options.mode.chained_assignment = None

#analysis_year = input("Enter current analysis year: ")
#data length can change based on start year of model
data_length = int(analysis_year)-1949

#need these file in working directory
df = pd.read_csv(data_dir+'\\'+str(analysis_year)+'_HTF_Annual_Outlook_full_data.csv',dtype={'stnID': str})
df['Station_ID'] = df.stnID.str[:7]
#annual historic enso data
enso_oni = pd.read_csv(f'{data_dir}/annual_means.csv')
#enso prediction
avg_file = pd.read_csv(f'{data_dir}/average_all_models.txt',header=None)
#add enso prediction to historical enso data
forecast_avg = avg_file.iloc[0][0]
enso_oni.loc[len(enso_oni)] = [int(analysis_year), np.nan, forecast_avg, np.nan]
enso_oni = enso_oni['ONI Met Year'].groupby(enso_oni['Year']).mean()
#read in station list with lat, lon, and station regions
lats_lons_df = pd.read_csv(stn_list)

#########################################################################################################

# functions
def quadratic_enso(x, z):
    return (a * x ** 2) + (b * x * z) + (c * x) + (d * z ** 2) + (e * z) + f
def quadratic(x):
    return (a * x ** 2) + (c * x) + f
def linear_enso(x,z):
    return (a * x) + (b*z) + c
def linear(x):
    return (a * x) + c
def no_t_lin_enso(z):
    return (a * z) + b
def no_t_quad_enso(z):
    return (a * z **2) + (b * z) + c

##########################################################################################################

for i in range(len(df)):
    
    #getting historical flood counts from API
    station = df.loc[i,'Station_ID']

    product = 'htf_met_year_annual'

    server = "https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/"

    myurl = (server + product +'.json?station='+station)

    urlResponse = requests.get(myurl)

    content=urlResponse.json()

    mydata = content['MetYearAnnualFloodCount']

    # Make it a Dataframe
    obs_df = pd.DataFrame(mydata)
    obs_df=obs_df[obs_df['metYear']>1949]
    if str(analysis_year) == str(2023):
        print('Removing 2023 observations')
        obs_df=obs_df[obs_df['metYear']<2023]

    obs = obs_df.dropna(subset=['minCount'])
    obs['HTF']='days per year'
    data_start_year = obs['metYear'].iloc[0]

    ################################################################################

    station_df = df.loc[df['Station_ID']==station]

    station_method = station_df['projectMethod']

    #for stations with a project method with ENSO sensitivity, we also want to grab the time-based method as well
    if station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
        trend_only = station_df[['Station_ID','stnName','lin_pred','lin_rmse']]
        trend_only['Low_Pred'] = trend_only['lin_pred'] - trend_only['lin_rmse']
        trend_only['High_Pred'] = trend_only['lin_pred'] + trend_only['lin_rmse']

    elif station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
        trend_only = station_df[['Station_ID','stnName','quad_pred','quad_rmse']]
        trend_only['Low_Pred'] = trend_only['quad_pred'] - trend_only['quad_rmse']
        trend_only['High_Pred'] = trend_only['quad_pred'] + trend_only['quad_rmse']

    elif station_df['projectMethod'].iloc[0] in ['No Temporal Trend with Linear ENSO Sensitivity', 'No Temporal Trend with Quadratic ENSO Sensitivity']:
        trend_only = station_df[['Station_ID','stnName','avg19','stdev19']]
        trend_only['Low_Pred'] = trend_only['avg19'] - trend_only['stdev19']
        if trend_only['Low_Pred'].iloc[0] < 0:
            trend_only['Low_Pred'] = 0
        trend_only['High_Pred'] = trend_only['avg19'] + trend_only['stdev19']

    predict_df = station_df[['Station_ID','stnName','lowConf','highConf']]
    predict_df.rename(columns={'lowConf':'Low_Pred','highConf':'High_Pred'},inplace=True)

    #for plotting outlook with historical data
    metYear=int(analysis_year)-0.25
    nextYear=int(analysis_year)+0.35

    try:
        trend_only['metYear']=metYear
        trend_only['nextYear']=nextYear
 
    except NameError:
        print('Not an ENSO station')

    predict_df['metYear']=metYear
    predict_df['nextYear']=nextYear
 
    #######################################################################################

    #getting trend formulas
    if station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
        station_formula = station_df['lin_enso_fit'].iloc[0]
        station_trend = station_df['lin_fit'].iloc[0]
        coefficients = ast.literal_eval(station_formula)
        coefficients_trend = ast.literal_eval(station_trend)
    elif station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
        station_formula = station_df['quad_enso_fit'].iloc[0]
        station_trend = station_df['quad_fit'].iloc[0]
        coefficients = ast.literal_eval(station_formula)
        coefficients_trend = ast.literal_eval(station_trend)
    elif station_df['projectMethod'].iloc[0] == 'Linear':
        station_formula = station_df['lin_fit'].iloc[0]
        coefficients = ast.literal_eval(station_formula)
    elif station_df['projectMethod'].iloc[0] == 'Quadratic':
        station_formula = station_df['quad_fit'].iloc[0]
        coefficients = ast.literal_eval(station_formula)
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Linear ENSO Sensitivity':
        station_formula = station_df['no_t_lin_fit'].iloc[0]
        station_trend = station_df['avg19'].iloc[0]
        coefficients = ast.literal_eval(station_formula)
        coefficients_avg = station_trend
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Quadratic ENSO Sensitivity':
        station_formula = station_df['no_t_quad_fit'].iloc[0]
        station_trend = station_df['avg19'].iloc[0]
        coefficients = ast.literal_eval(station_formula)
        coefficients_avg = station_trend
    else:
        station_formula = station_df['avg19']
        coefficients_avg = station_formula

    #######################################################################################

    #calculating trend lines
    x_vals = np.linspace(1950, int(analysis_year),data_length)
    z = 0
    
        # Extract coefficients
    if station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
        a = coefficients['np.power(x, 2)']
        b = coefficients['x:z']
        c = coefficients['x']
        d = coefficients['np.power(z, 2)']
        e = coefficients['z']
        f = coefficients['Intercept']
        y_vals = quadratic_enso(x_vals, z)
        y_vals_trend = quadratic_enso(x_vals,enso_oni)
        other_method = 'Quadratic'
    elif station_df['projectMethod'].iloc[0] =='Quadratic':
        a = coefficients['np.power(x, 2)']
        c = coefficients['x']
        f = coefficients['Intercept']
        y_vals_trend = quadratic(x_vals)
    elif station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
        a = coefficients['x']
        b = coefficients['z']
        c = coefficients['Intercept']
        y_vals = linear_enso(x_vals, z)
        y_vals_trend = linear_enso(x_vals,enso_oni)
        other_method = 'Linear'
    elif station_df['projectMethod'].iloc[0] == 'Linear':
        a = coefficients['x']
        c = coefficients['Intercept']
        y_vals_trend = linear(x_vals)
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Linear ENSO Sensitivity':
        a = coefficients['z']
        b = coefficients['Intercept']
        y_vals_trend = no_t_lin_enso(enso_oni)
        y_vals = [coefficients_avg]*data_length
        other_method = '19-yr Average'
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Quadratic ENSO Sensitivity':
        a = coefficients['np.power(z, 2)']
        b = coefficients['z']
        c = coefficients['Intercept']
        y_vals_trend = no_t_quad_enso(enso_oni)
        y_vals = [coefficients_avg]*data_length
        other_method = '19-yr Average'
    else:
        y_vals_trend = [coefficients_avg]*data_length

    if station == '1820000':
        max_y_lim = round(obs_df['minCount'].max()/5)*5+15
    else:
        max_y_lim = round(obs_df['minCount'].max()/5)*5+10

    #######################################################################################

    plt.figure(figsize=(16, 8))
    bar_width = 0.7
    plt.bar(obs['metYear'], obs['minCount'], color='lightskyblue',width=bar_width,label = 'Historical HTF days')

    if station_df['projectMethod'].iloc[0] in ['Linear With ENSO Sensitivity','Quadratic With ENSO Sensitivity','No Temporal Trend with Linear ENSO Sensitivity','No Temporal Trend with Quadratic ENSO Sensitivity']:
        plt.fill_between(trend_only['metYear'], trend_only['Low_Pred'], trend_only['High_Pred'], color='black', step='mid', linewidth=7, label=other_method)
        plt.fill_between(predict_df['nextYear'], predict_df['Low_Pred'], predict_df['High_Pred'], color='red', step='mid', linewidth=7, label=station_df['projectMethod'].iloc[0])
        plt.plot(x_vals, y_vals, color='black', linewidth=2,alpha=0.5)
        plt.plot(x_vals, y_vals_trend, color='red', linewidth=2,alpha=0.5)
    else:
        plt.fill_between(predict_df['nextYear'], predict_df['Low_Pred'], predict_df['High_Pred'], color='black', step='mid', linewidth=7, label=station_df['projectMethod'].iloc[0])
        plt.plot(x_vals, y_vals_trend, color='black', linewidth=2,alpha=0.5)
    # Set labels and title
    plt.title('2024-25 Annual HTF Projection with Historical Context\n'+obs_df.iloc[0]['stnId']+' '+obs_df.iloc[0]['stnName'])

    plt.xlabel('Years')
    plt.ylabel('Flood Count (days/yr)')
    plt.xlim(data_start_year-2, 2025)   # Adjust x-axis limits if needed
    plt.ylim(0,max_y_lim)   # Adjust y-axis limits as requested
    plt.legend(loc='upper left')
    fig_path=f'{data_dir}\\{analysis_year}_long_term_plots/'
    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    plt.savefig(fname=fig_path+station+'.png')
  
print('Station graphs done')
#########################################################################################

#map 1
#Upper range of next year's prediction 
lats_lons_df['NOAA_ID'] = lats_lons_df['NOAA_ID'].astype(str)
df = pd.merge(df,lats_lons_df,left_on='Station_ID',right_on='NOAA_ID')

df['mid_pred'] = (df['highConf']+df['lowConf'])/2
lats = list(df['Lat'])
lons = list(df['Lon'])
upper_pred_range = df['highConf']

min_val=df['highConf'].min()
max_val=df['highConf'].max()

#changed from top 10 to over 30 HTF days
top_10=df[df['highConf']>30]
upper_pred_range_10 = top_10['highConf']
lats_10 = list(top_10['Lat'])
lons_10 = list(top_10['Lon'])
names_10 = list(top_10['stnName'])

#######################################################################################

upper_pred_range_10 = upper_pred_range_10.reset_index(drop=True)
# Function to discretize values into 5-day increments
#def discretize_values(values, increment=10):
 #   return increment * (values // increment) +10
def discretize_values(values):
    # Create an array to hold discretized values
    discretized = np.zeros_like(values)
    
    # Set values over 30 to a special category (e.g., 31)
    discretized[values > 30] = 35
    
    # For values 0-30, discretize into 5-day increments
    bins = np.arange(0, 31, 5)
    discretized[values <= 30] = np.digitize(values[values <= 30], bins) * 5
    
    return discretized
# Discretize the upper prediction range into 5-day increments
discretized_upper_pred_range = discretize_values(upper_pred_range)
discretized_upper_pred_range_10 = discretize_values(upper_pred_range_10)

# Plotting
fig = plt.figure(figsize=(16, 14), dpi=100)
# Define Lambert Conformal projection parameters
lon_0 = -96  # Central longitude
lat_0 = 37   # Central latitude
lat_1 = 33   # First standard parallel
lat_2 = 45   # Second standard parallel (optional)
ax1=plt.axes()
ax2= plt.axes([0.12, 0.26, 0.16, 0.14])
ax3= plt.axes([0.22, 0.26, 0.25, 0.09])
ax1.set_title('Upper range of 2024-25 Outlook Predictions',size=20)
map1=Basemap(ax=ax1,llcrnrlon=-122, llcrnrlat=17, urcrnrlon=-48.,urcrnrlat=47.,lon_0 = -96,lat_0 = 37,lat_1 = 33,projection='lcc',resolution='i')
#map1.bluemarble()
map1.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map1.drawcoastlines(zorder=2)
map1.drawcountries(zorder=3)
map1.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map1(lons, lats)
x_10, y_10 = map1(lons_10, lats_10)

# Colorbar
# Use discrete color bins and colormap
cmap = plt.cm.get_cmap('Reds', (discretized_upper_pred_range.max() // 5))
norm = plt.Normalize(vmin=0, vmax=discretized_upper_pred_range.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = map1.colorbar(sm, shrink=0.8)
cbar.set_label('Number of HTF Days/Year', fontsize=17)

sc = map1.scatter(x, y, c=discretized_upper_pred_range, cmap=cmap, norm=norm,s=40,zorder=9)
marker_sizes = 100 + upper_pred_range_10 * 10  # Adjust size based on HTF days
sc_10 =map1.scatter(x_10, y_10, c=discretized_upper_pred_range_10, cmap=cmap,norm=norm, s=marker_sizes, marker='*',edgecolor='red', label='Stations with an upper range of over 30 HTF days ', zorder=10)

# Set colorbar ticks and labels correctly
tick_positions = np.arange(discretized_upper_pred_range.min(), discretized_upper_pred_range.max() + 1, 5)
tick_labels = [f'{tick}' if tick < 30 else '30+' for tick in tick_positions]
# Remove the top tick

cbar.set_ticks(tick_positions)
cbar.set_ticklabels(tick_labels)

cbar.set_ticks(tick_positions[:-1])

# Pacific Region map
ax2.set_title('West Pacific Islands', size=10,color='black',backgroundcolor = 'white')
map2 = Basemap(ax=ax2, llcrnrlon=140., llcrnrlat=-17., urcrnrlon=192., urcrnrlat=31., projection='cea')
map2.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map2.drawcoastlines(zorder=2)
map2.drawcountries(zorder=3)
map2.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map2(lons, lats)
x_10, y_10 = map2(lons_10, lats_10)

# Use discrete color bins and colormap
map2.scatter(x, y, c=discretized_upper_pred_range, cmap=cmap,norm=norm, s=40,zorder=9)
marker_sizes = 100 + upper_pred_range_10 * 10  # Adjust size based on HTF days
map2.scatter(x_10, y_10, c=discretized_upper_pred_range_10, cmap=cmap, s=marker_sizes, norm=norm,marker='*', edgecolor='red', label='', zorder=10)

ax3.set_title('Hawaiian Islands', size=10, color='black',backgroundcolor = 'white')
map3 = Basemap(ax=ax3, llcrnrlon=198., llcrnrlat=17., urcrnrlon=207., urcrnrlat=23., projection='cea')
map3.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map3.drawcoastlines(zorder=2)
map3.drawcountries(zorder=3)
map3.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map3(lons, lats)
x_10, y_10 = map3(lons_10, lats_10)

# Use discrete color bins and colormap
map3.scatter(x, y, c=discretized_upper_pred_range, cmap=cmap,norm=norm, s=40,zorder=9)
marker_sizes = 100 + upper_pred_range_10 * 10  # Adjust size based on HTF days
map3.scatter(x_10, y_10, c=discretized_upper_pred_range_10, cmap=cmap, s=marker_sizes, norm=norm,marker='*', edgecolor='red', label='', zorder=10)

# Adjust spines and legend'
for spine in ax2.spines.values():
    spine.set_color('red')
for spine in ax3.spines.values():
    spine.set_color('red')
plt.legend(loc='upper right', fontsize=10, edgecolor='white')
plt.savefig(fname=f'{data_dir}\\{analysis_year}_map_upper_range_htf_annual_outlook.png')

#save complementary csv
upper_pred_range_download = df[['Station_ID' ,'Lat','Lon','highConf',]]

# Determine top five stations based on some criteria (e.g., highest number of HTF days)
top_indices = df[df['highConf']> 29].index  # Get indices of top five stations
top_flags = np.zeros(len(upper_pred_range))         # Initialize flag array
top_flags[top_indices] = 1

# Replace 1 with 'yes' and 0 with 'no' in a new column 'top_five'
upper_pred_range_download['30_plus_days'] = np.where(top_flags == 1, 'YES', 'NO')
upper_pred_range_download.to_csv(f'{data_dir}\\{analysis_year}_map_upper_range_htf_annual_outlook.csv')

print('Map 1 done')

#################################################################################################

#map 2
#Number of days increase since 2000
df['diff_2000'] = df['mid_pred'] - df['pred_2k']

diff_2000 = df['diff_2000']

min_val=df['diff_2000'].min()
max_val=df['diff_2000'].max()

lats = list(df['Lat'])
lons = list(df['Lon'])

top_10=df[df['diff_2000']>10]
diff_2000_10 = top_10['diff_2000']

lats_10 = list(top_10['Lat'])
lons_10 = list(top_10['Lon'])

###############################################################################################

diff_2000_10 = diff_2000_10.reset_index(drop=True)
# Function to discretize values into 5-day increments
#def discretize_values(values, increment=10):
 #   return increment * (values // increment) +10
def discretize_values(values):
    # Create an array to hold discretized values
    discretized = np.zeros_like(values)
    
    # Set values over 30 to a special category (e.g., 31)
    discretized[values > 30] = 35
    
    # For values 0-30, discretize into 5-day increments
    bins = np.arange(0, 31, 5)
    discretized[values <= 30] = np.digitize(values[values <= 30], bins) * 5
    
    return discretized
# Discretize the upper prediction range into 5-day increments
discretized_diff_2000 = discretize_values(diff_2000)
discretized_diff_2000_10 = discretize_values(diff_2000_10)

# Plotting
fig = plt.figure(figsize=(16, 14), dpi=100)
# Define Lambert Conformal projection parameters
lon_0 = -96  # Central longitude
lat_0 = 37   # Central latitude
lat_1 = 33   # First standard parallel
lat_2 = 45   # Second standard parallel (optional)
ax1=plt.axes()
ax2= plt.axes([0.12, 0.26, 0.16, 0.14])
ax3= plt.axes([0.22, 0.26, 0.25, 0.09])
ax1.set_title('2024-25 Predicted Increases in High Tide Flooding since 2000',size=20)
map1=Basemap(ax=ax1,llcrnrlon=-122, llcrnrlat=17, urcrnrlon=-48.,urcrnrlat=47.,lon_0 = -96,lat_0 = 37,lat_1 = 33,projection='lcc',resolution='i')
#map1.bluemarble()
map1.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map1.drawcoastlines(zorder=2)
map1.drawcountries(zorder=3)
map1.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map1(lons, lats)
x_10, y_10 = map1(lons_10, lats_10)

# Use discrete color bins and colormap
cmap = plt.cm.get_cmap('Reds', (discretized_diff_2000.max() // 5))
sc = map1.scatter(x, y, c=diff_2000, cmap=cmap, norm=norm,s=40,zorder=9)
marker_sizes = 100 + diff_2000_10 * 10  # Adjust size based on HTF days
sc_10 =map1.scatter(x_10, y_10, c=diff_2000_10, cmap=cmap,norm=norm, s=marker_sizes, marker='*',edgecolor='red', label='Stations with over 10+ days of increased HTF days since 2000', zorder=10)

# Colorbar
norm = plt.Normalize(vmin=0, vmax=discretized_diff_2000.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = map1.colorbar(sm, shrink=0.8)
cbar.set_label('Number of HTF Days/Year', fontsize=17)

# Set colorbar ticks and labels correctly
tick_positions = np.arange(discretized_diff_2000.min(), discretized_diff_2000.max(), 5)
tick_labels = [f'{tick}' if tick < 30 else '30+' for tick in tick_positions]
# Remove the top tick

cbar.set_ticks(tick_positions)
cbar.set_ticklabels(tick_labels)

# Pacific Region map
ax2.set_title('West Pacific Islands', size=10,color='black',backgroundcolor = 'white')
map2 = Basemap(ax=ax2, llcrnrlon=140., llcrnrlat=-17., urcrnrlon=192., urcrnrlat=31., projection='cea')
map2.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map2.drawcoastlines(zorder=2)
map2.drawcountries(zorder=3)
map2.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map2(lons, lats)
x_10, y_10 = map2(lons_10, lats_10)

# Use discrete color bins and colormap
map2.scatter(x, y, c=diff_2000, cmap=cmap,norm=norm, s=40,zorder=9)
marker_sizes = 100 + diff_2000_10 * 10  # Adjust size based on HTF days
map2.scatter(x_10, y_10, c=diff_2000_10, cmap=cmap, s=marker_sizes, norm=norm,marker='*', edgecolor='red', label='', zorder=10)

ax3.set_title('Hawaiian Islands', size=10, color='black',backgroundcolor = 'white')
map3 = Basemap(ax=ax3, llcrnrlon=198., llcrnrlat=17., urcrnrlon=207., urcrnrlat=23., projection='cea')
#map2.bluemarble()
map3.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map3.drawcoastlines(zorder=2)
map3.drawcountries(zorder=3)
map3.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map3(lons, lats)
x_10, y_10 = map3(lons_10, lats_10)

# Use discrete color bins and colormap
map3.scatter(x, y, c=diff_2000, cmap=cmap,norm=norm, s=40,zorder=9)
marker_sizes = 100 + diff_2000_10 * 10  # Adjust size based on HTF days
map3.scatter(x_10, y_10, c=diff_2000_10, cmap=cmap, s=marker_sizes, norm=norm,marker='*', edgecolor='red', label='', zorder=10)

# Adjust spines and legend'
for spine in ax2.spines.values():
    spine.set_color('red')
for spine in ax3.spines.values():
    spine.set_color('red')
plt.legend(loc='upper right', fontsize=10, edgecolor='white')
plt.savefig(fname=f'{data_dir}\\{analysis_year}_map_increase_htf_annual_outlook.png')

#save complementary csv
diff_2000_download = df[['Station_ID' ,'Lat','Lon','diff_2000','mid_pred','pred_2k']]

# Determine top five stations based on some criteria (e.g., highest number of HTF days)
top_indices = df[df['diff_2000']> 10].index   # Get indices of top five stations
top_flags = np.zeros(len(diff_2000_download))         # Initialize flag array
top_flags[top_indices] = 1

# Replace 1 with 'yes' and 0 with 'no' in a new column 'top_five'
diff_2000_download['10_plus_days'] = np.where(top_flags == 1, 'YES', 'NO')
diff_2000_download.to_csv(f'{data_dir}\\{analysis_year}_map_increase_htf_annual_outlook.csv')
print('Map 2 done')

################################################################################################

#map 3
#Difference in number of days between next year's trend-only prediction and next year's trend+ENSO
enso_diff = pd.DataFrame(columns=['diff_enso'])

for i in range(len(df)):
    station = df.loc[i,'Station_ID']
    station_df = df.loc[df['Station_ID']==station]
    
    if station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
        x = station_df['lin_enso_pred'].iloc[0] - station_df['lin_pred'].iloc[0]
        enso_diff = pd.concat([enso_diff, pd.DataFrame({'diff_enso': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
        x = station_df['quad_enso_pred'].iloc[0] - station_df['quad_pred'].iloc[0]
        enso_diff = pd.concat([enso_diff, pd.DataFrame({'diff_enso': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Linear':
        x = np.nan
        enso_diff = pd.concat([enso_diff, pd.DataFrame({'diff_enso': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Quadratic':
        x = np.nan
        enso_diff = pd.concat([enso_diff, pd.DataFrame({'diff_enso': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Linear ENSO Sensitivity':
        x = station_df['no_t_lin_pred'].iloc[0] - station_df['avg19'].iloc[0]
        enso_diff = pd.concat([enso_diff, pd.DataFrame({'diff_enso': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Quadratic ENSO Sensitivity':
        x = station_df['no_t_quad_pred'].iloc[0] - station_df['avg19'].iloc[0]
        enso_diff = pd.concat([enso_diff, pd.DataFrame({'diff_enso': [x]})], ignore_index=True)
        
    else:
        x = np.nan
        enso_diff = pd.concat([enso_diff, pd.DataFrame({'diff_enso': [x]})], ignore_index=True)
df['diff_enso'] = enso_diff
df_enso = df.dropna()
lats = list(df_enso['Lat'])
lons = list(df_enso['Lon'])

diff_enso = df_enso['diff_enso']

min_val=df['diff_enso'].min()
max_val=df['diff_enso'].max()

##########################################################################################

def discretize_values(values):
    max_val=4
   # Create an array to hold discretized values
    discretized = np.zeros_like(values, dtype=int)
    
    # For values less than -5, set to -5
    discretized[values < -5] = -5
    
    # For values between -5 and 5 (inclusive), discretize into 1-day increments
    discretized[(values >= -5) & (values < 0)] = np.floor(values[(values >= -5) & (values < 0)])
    discretized[(values <= 5) & (values > 0)] = np.ceil(values[(values<= 5) & (values > 0)])
    # For values greater than 5, set to 10 (or any suitable max value)
    discretized[values > 5] = 10
    
    return discretized
    
# Discretize the upper prediction range into 5-day increments
discretized_diff_enso = discretize_values(diff_enso)

# Plotting
fig = plt.figure(figsize=(16, 14), dpi=100)
# Define Lambert Conformal projection parameters
lon_0 = -96  # Central longitude
lat_0 = 37   # Central latitude
lat_1 = 33   # First standard parallel
lat_2 = 45   # Second standard parallel
ax1=plt.axes()
ax2= plt.axes([0.12, 0.26, 0.16, 0.14])
ax3= plt.axes([0.22, 0.26, 0.25, 0.09])
ax1.set_title('Impact of ENSO on Predicted Annual Outlook\n\nCalculated as: chosen ENSO+trend method - equivalent trend-only method',size=20)
#plt.title('Impact of ENSO on Predicted Annual Outlook',size=20)
#plt.suptitle('Calculated as chosen ENSO method - equivalent trend only method',size=15)
map1=Basemap(ax=ax1,llcrnrlon=-122, llcrnrlat=17, urcrnrlon=-48.,urcrnrlat=47.,lon_0 = -96,lat_0 = 37,lat_1 = 33,projection='lcc',resolution='i')
#map1.bluemarble()
map1.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map1.drawcoastlines(zorder=2)
map1.drawcountries(zorder=3)
map1.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map1(lons, lats)
x_10, y_10 = map1(lons_10, lats_10)

# Use discrete color bins and colormap
# Determine the number of bins for the colormap
vmin = -5
vmax = 5
num_bins = (vmax - vmin)
cmap = plt.cm.get_cmap('seismic',num_bins)

# Colorbar
norm = plt.Normalize(vmin, vmax)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = map1.colorbar(sm, shrink=0.8)
cbar.set_label('Difference in Number of HTF Days/Year', fontsize=17)
sc = map1.scatter(x, y, c=diff_enso, cmap=cmap, norm=norm,s=40,zorder=9)
#marker_sizes = 100 + diff_2000_10 * 10  # Adjust size based on HTF days
#sc_10 =map1.scatter(x_10, y_10, c=diff_2000_10, cmap=cmap,norm=norm, s=marker_sizes, marker='*',edgecolor='red', label='Stations with over 10+ days of increased HTF days since 2000', zorder=10)

# Annotate station names and upper predictions for top 10 stations
'''for i, (lon, lat) in enumerate(zip(x_10, y_10)):
    ax1.annotate(f'{i+1}', (lon, lat), fontsize=20, color='blue',zorder=12)'''


# Set colorbar ticks and labels correctly
#tick_positions = np.arange(discretized_upper_pred_range.min(), discretized_upper_pred_range.max(),5)
#tick_labels = np.arange(vmin, vmax+1, 1)
tick_positions = np.arange(vmin, vmax+1, 1)
#tick_positions = np.arange(discretized_diff_2000.min(), discretized_diff_2000.max(), 5)
tick_labels = [f'{tick}' if tick < 5 else '5+' for tick in tick_positions]
# Remove the top tick

cbar.set_ticks(tick_positions)
#
cbar.set_ticklabels(tick_labels)

#cbar.set_ticks(tick_positions[:-1])

# Pacific Region map
#ax2 = fig.add_axes([0.55, 0.1, 0.35, 0.4])  # Adjust position as needed
ax2.set_title('West Pacific Islands', size=10,color='black',backgroundcolor = 'white')
map2 = Basemap(ax=ax2, llcrnrlon=140., llcrnrlat=-17., urcrnrlon=192., urcrnrlat=31., projection='cea')
#map2.bluemarble()
map2.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map2.drawcoastlines(zorder=2)
map2.drawcountries(zorder=3)
map2.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map2(lons, lats)
x_10, y_10 = map2(lons_10, lats_10)

# Use discrete color bins and colormap
map2.scatter(x, y, c=diff_enso, cmap=cmap,norm=norm, s=40,zorder=9)
#marker_sizes = 100 + diff_2000_10 * 10  # Adjust size based on HTF days
#map2.scatter(x_10, y_10, c=diff_2000_10, cmap=cmap, s=marker_sizes, norm=norm,marker='*', edgecolor='red', label='', zorder=10)
'''for i, (lon, lat) in enumerate(zip(x_10, y_10)):
    ax2.annotate(f'{i}', (lon, lat), fontsize=10, color='red',xytext=(lon, lat),
            arrowprops=dict(facecolor='gray', shrink=0.05),zorder=12)'''
ax3.set_title('Hawaiian Islands', size=10, color='black',backgroundcolor = 'white')
map3 = Basemap(ax=ax3, llcrnrlon=198., llcrnrlat=17., urcrnrlon=207., urcrnrlat=23., projection='cea',resolution='i')
#map2.bluemarble()
map3.drawlsmask(land_color='dimgray', ocean_color='k', lakes=True,zorder=1)
map3.drawcoastlines(zorder=2)
map3.drawcountries(zorder=3)
map3.fillcontinents(color='dimgray', lake_color='k',zorder=4)

x, y = map3(lons, lats)
x_10, y_10 = map3(lons_10, lats_10)

# Use discrete color bins and colormap
map3.scatter(x, y, c=diff_enso, cmap=cmap,norm=norm, s=40,zorder=9)
#marker_sizes = 100 + diff_2000_10 * 10  # Adjust size based on HTF days
#map3.scatter(x_10, y_10, c=diff_2000_10, cmap=cmap, s=marker_sizes, norm=norm,marker='*', edgecolor='red', label='', zorder=10)

# Adjust spines and legend'
for spine in ax2.spines.values():
    spine.set_color('red')
for spine in ax3.spines.values():
    spine.set_color('red')

plt.savefig(fname=f'{data_dir}\\{analysis_year}_map_diff_enso_htf_annual_outlook.png')

diff_enso_download = df[['Station_ID' ,'Lat','Lon','diff_enso','projectMethod']]
diff_enso_download.to_csv(f'{data_dir}\\{analysis_year}_map_diff_enso_htf_annual_outlook.csv')

print('Map 3 done')

###############################################################################################

#regional bar plot
#try different method showing all data points in histogram
conus_list=['NE','MID','SE','WGC','EGC','NW','SW']
df_conus=df[df['Region'].isin(conus_list)]
df_conus=df_conus[['mid_pred']]
df_conus=df_conus.to_numpy()

df_ne=df[df['Region']=='NE']
df_ne=df_ne[['mid_pred']]
df_ne=df_ne.to_numpy()

df_mid=df[df['Region']=='MID']
df_mid=df_mid[['mid_pred']]
df_mid=df_mid.to_numpy()

df_se=df[df['Region']=='SE']
df_se=df_se[['mid_pred']]
df_se=df_se.to_numpy()

df_car=df[df['Region']=='CAR']
df_car=df_car[['mid_pred']]
df_car=df_car.to_numpy()

df_wgc=df[df['Region']=='WGC']
df_wgc=df_wgc[['mid_pred']]
df_wgc=df_wgc.to_numpy()

df_egc=df[df['Region']=='EGC']
df_egc=df_egc[['mid_pred']]
df_egc=df_egc.to_numpy()

df_nw=df[df['Region']=='NW']
df_nw=df_nw[['mid_pred']]
df_nw=df_nw.to_numpy()

df_sw=df[df['Region']=='SW']
df_sw=df_sw[['mid_pred']]
df_sw=df_sw.to_numpy()

df_pac=df[df['Region']=='PAC']
df_pac=df_pac[['mid_pred']]
df_pac=df_pac.to_numpy()

data = [df_conus.flatten(),df_ne.flatten(),df_mid.flatten(),df_se.flatten(),df_car.flatten(),df_wgc.flatten(),
        df_egc.flatten(),df_nw.flatten(),df_sw.flatten(),df_pac.flatten()]
labels = ['CONUS','NE','MID','SE','CAR','WGC','EGC','NW','SW','PAC']
category_colors = ['#970C10','#1E73BE']

################################################################################

f1 = plt.figure(figsize=(10,7))
ax1=f1.add_subplot(111)

plt.boxplot(data,patch_artist=True,labels=labels)
ax1.set_title(str(analysis_year)+"2024-25 HTF Annual Outlook by Region")
ax1.set_ylabel('HTF days/year')
plt.yticks(np.arange(0, 100, step=10))
plt.grid()
f1.savefig(fname=f'{data_dir}\\{analysis_year}_regional_htf_annual_outlook.png')
regional_boxplot_download = df[['Station_ID' ,'Lat','Lon','Region','mid_pred']]
regional_boxplot_download.to_csv(f'{data_dir}\\{analysis_year}_regional_htf_annual_outlook.csv')

print('Regions graph done')
print('End Script')