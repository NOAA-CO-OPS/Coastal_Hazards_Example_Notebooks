import pandas as pd
import datetime as dt
import os
import urllib3
from io import StringIO
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
run_year = data_params['run_year']


# Get ENSO data from CPC via URL

http = urllib3.PoolManager()

response = http.request('GET','https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt')
cpc_oni_resp = response.data.decode('utf-8')
cpc_oni_data = StringIO(cpc_oni_resp)
cpc_oni_df = pd.read_csv(cpc_oni_data,sep='\s+')
cpc_df_len = len(cpc_oni_df.index)
# Data Columns - YR, MON, TOTAL, ClimAdjust, ANOM

anom_list = []

for index, row in cpc_oni_df.iterrows():
    anom = float(cpc_oni_df.loc[index,'ANOM'])
    if index == 0:
        anom_list.append((anom+float(cpc_oni_df.loc[index+1,'ANOM']))/2)
    elif index == cpc_df_len-1:
        anom_list.append((anom+float(cpc_oni_df.loc[index-1,'ANOM']))/2)
    else:
        anom_list.append((anom+float(cpc_oni_df.loc[index-1,'ANOM'])+float(cpc_oni_df.loc[index+1,'ANOM']))/3)

cpc_oni_df['ONI'] = anom_list

year_list = cpc_oni_df['YR'].unique().tolist()

#can uncomment if wanting to download this info
#cpc_oni_df.to_csv(f'{data_dir}\\enso_oni.csv',index=False)

# Saves ONI data as a csv
#print(f"CPC ONI CSV created as {data_dir}\\enso_oni.csv\n")

annual_means = pd.DataFrame(columns=['Year', 'ONI Calendar Year', 'ONI Met Year', 'ONI Cool Season'])

# Calculate means by calendar year (Jan-Dec), met year (May-Apr), and NH 'cool' season (Nov - Mar)

for year in year_list:
    if year<int(run_year):
        cal_index = cpc_oni_df[cpc_oni_df.YR == year].index.tolist()[0]
        met_index = cal_index+4
        nh_index = cal_index+10
        cal_df = cpc_oni_df[cal_index:cal_index+12]
        met_df = cpc_oni_df[met_index:met_index+12]
        nh_df = cpc_oni_df[nh_index:nh_index+5]
        cal_mean = round(cal_df['ONI'].mean(),1)
        met_mean = round(met_df['ONI'].mean(),1)
        nh_mean = round(nh_df['ONI'].mean(),1)
        temp_rec = pd.DataFrame(data=[year,cal_mean,met_mean,nh_mean]).transpose()
        temp_rec.columns = annual_means.columns
        annual_means = pd.concat([annual_means,temp_rec])
        #annual_means[year]= [cal_mean,met_mean,nh_mean]

# Saves means into a CSV file

annual_means.to_csv(f'{data_dir}\\annual_means.csv',index=False)

print(f"Annual Means CSV created as {data_dir}\\annual_means.csv")
