import pandas as pd
import requests # need to install lxml as well
import datetime as dt
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
run_year = data_params['run_year']


# Import and Derive Forecast Plume for Upcoming Meteorological Year and Calculate the Average of All Model Averages

# Important to note that the website for the June outlook changes when the July is released (which then becomes the 'current' site)
# This should pull the June forecast website, unless it doesn't exist yet, in which case it will pull the most current one (which should be May if run in late May/early June, or June if run in late June/early July)

try:
    enso_html = requests.get(f'https://iri.columbia.edu/our-expertise/climate/forecasts/enso/{run_year}-June-quick-look/?enso_tab=enso-sst_table').content
    enso_table = pd.read_html(enso_html)
except:
    print(f"Seperate URL for June {run_year} does not exist; using URL for most current forecast (which may be the June {run_year} if the July forecast has not been published yet)")
    enso_html = requests.get('https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/?enso_tab=enso-sst_table').content
    enso_table = pd.read_html(enso_html)

for df in enso_table:
    if df.iloc[0,0]=='Dynamical Models': # Find Dynamical Models Table
        print(df.iloc[-1,1:])
        model_avg = df.iloc[-1,1:].values # Find last element (Average of all models)
        avg_all = round(model_avg.astype(float).mean(),1) 
        break

# Saves the average of all models in a text file

avg_file = open(f"{data_dir}\\average_all_models.txt","w+")

avg_file.write(str(avg_all))

avg_file.close()

print(f"Average of all models created as {data_dir}\\average_all_models.txt")
