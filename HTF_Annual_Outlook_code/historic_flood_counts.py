import pandas as pd
import requests
import datetime as dt
import os
import  configparser

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

# Obtain historical High Tide Flooding by Met Year from NOAA COOPS API

met_yr_url = 'https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/htf_met_year_annual.xml'

met_yr_resp = requests.get(met_yr_url).content

met_yr_df = pd.read_xml(met_yr_resp)

met_yr_df = met_yr_df.drop(columns=['count'])

# Drops any rows with no station ID

met_yr_df = met_yr_df.dropna(subset=['stnId'])

met_yr_df = met_yr_df.reset_index(drop=True)

# Saves HTF counts as a csv 

met_yr_df.to_csv(f'{data_dir}\\met_historic_flood_cts.csv',index=False)

print(f"Historic Flood Counts CSV created as {data_dir}\\met_historic_flood_cts.csv")
