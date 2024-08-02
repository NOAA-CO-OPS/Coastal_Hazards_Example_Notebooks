# Annual High Tide Flooding Python Code

# Overview

The [Annual High Tide Flooding (HTF) Outlook](https://tidesandcurrents.noaa.gov/high-tide-flooding/annual-outlook.html) is NOAA’s product that provides the number of predicted high tide flooding days for the upcoming meteorological year (May to April). This repository contains Python code used to produce NOAA’s Annual High Tide Flooding (HTF) Outlook. This code which was originally written in MatLab has been translated to Python to streamline and automate the process of producing Annual HTF Outlooks, and provide broader, open-source access to authoritative data sources. This suite of code was first used to develop the 2024-25 Annual HTF Outlook, and is the operational source code for future Annual HTF Outlooks until methods are modified or enhanced further. It also contains scripts to generate station-based and regional-based statistics as well as graphics. In addition, there is an interactive notebook in which the user can explore the underlying data and results.

## ***Set Up***
To run the code, users will need to set up a virtual environment with certain modules. The ht_outlook.yml file provides the conda environment needed to run all of the scripts in this project. 

Simply create the environment using this command in your terminal:
conda env create -f ht_outlook.yml

And activate the environment:
conda activate ht_outlook

The user will also need to make changes to the configuration file for the code, as detailed in the next section.

## ***Modifying the Configuration aFile***:
The configuration file for the code will be downloaded along the rest of the code, and is named ‘config.cfg’. Two parameters must be changed, while changing the others is optional.
Change the Run Year: 

The run_year parameter must be changed to the current year. This ensures the code runs on the full time period up to the current year. 

Change the Station List Name:

The name of the station list file must be defined here. If the file is included in the same folder as the code, only the station name is needed (i.e., stn_list_file = HTF_Annual_Outlook_StationList_2024.csv). The NOAA station ID and its latitude and longitude are the needed fields from the station list.

If the station file is located in a different folder than the code, the full file path must be included along with the file name (i.e., stn_list_file = C:\HTF_Annual_Outlook_StationList_2024.csv)

Optional: Change Directories:
If the default directories are not created and a different set of directories are going to be used, the work_dir, run_dir, and data_dir parameters should be changed to match the desired directory paths. Make sure to create the desired directory before running the code.

Optional: Change Other Parameters:

The start_year parameter defaults to 1950, which defines how far back historical counts will be retrieved for regression fits. The pvalue_limit parameter defaults to 0.101, which defines the upper limit for the p-value of the regression fits that choose which method is used to predict next year’s historical counts. Both of these values can be changed but should be kept at the default unless directed otherwise or for testing purposes.  

## ***Running the Prerequisite Scripts***
Three scripts need to be run before the main Annual High Tide Outlook code. Each of these scripts downloads and stores a different set of data that is used as an input for the main code. These can be run separately and do not need to be run immediately prior to the main code, but generally should not be run until certain dates, noted in detail below. Additionally, the configuration file parameters should not be changed between running these scripts and the main code. The main code relies on using the same parameters to know where to ingest the data downloaded by these scripts. 

Once the configuration file is set up correctly and the user is in the environment set up in step II, all of these scripts can be run using ‘python <script name>’  in the folder with the code.

The ***ENSO Forecast Average***
The script enso_forecast_avg.py downloads the average of models for the ENSO forecast of the upcoming meteorological year, which will be used to calculate the predicted range of flooding values for the next year. This should generally only be run once the June forecast is released. It can be prior to this, but it will instead download the average from the most recent month’s forecast which typically should not be used for the Annual outlook. 

## ***ENSO ONI Annual Means***
The script enso_oni_download.py downloads the ONI values for each month for the period specified by the configuration file, and averages them by calendar year, meteorological year, and the ‘cool’ season of November to March. This data is necessary to incorporate ENSO into the regression fit methods. The meteorological year averages are used as an input for the main code, so this script should only be run after April to get the full data for the last meteorological year. 

## ***Historic Flood Counts***
The script historic_flood_counts.py downloads the flood counts for every meteorological year over the time period specified in the configuration file for every station in the station list. This data is critical to develop the regression that will be used to predict the next year’s flooding days. This module should only be run once CO-OPS water level data is processed and verified hourly water levels for April of the current year, typically completed by the end of May. 

## ***Annual High Tide Outlook Python Prediction Script***
The script regression_fit.py calculates regression coefficients to fit historical data. It also calculates p-values for the fitness of these coefficients and chooses one regression model based on a decision tree to calculate the predicted values for the next year. This script should only be run after the three scripts in part III are run, as it requires this information to create the predictions. The script can be run with the command ‘python regression_fit.py’ in the same folder as the code. 
The script creates two CSV files, which are prepended by the run year to distinguish between different annual runs:
HTF_Annual_Outlook.csv: The primary output which summarizes the method chosen and the range of predicted days for each station
HTF_Annual_Outlook_full_data.csv: A more detailed output that includes all of the coefficients for each method as well as the p-values and RMSEs for each. This is primarily used for the graphics and statistics scripts but can also be used to better understand why a method was or was not chosen.
Generally, only the first file is utilized as part of the final results for the Annual High Tide Outlook. The full data file instead enables the next step of the process, running the graphics and statistics scripts. These files will be saved to the directory specified in the configuration file in the data_dir parameter. 

## ***HTF Annual Outlook Graphics.py*** 
This script HTF_annual_outlook_graphics.py takes the results from the regression_fit.py and outputs graphics. The first plot is a long-term station plot containing historical annual observations (blue bars), historical trend (black line), historical trend with ENSO sensitivity (for applicable stations) (red line), next year's prediction from historical trend (black bar), and next year's prediction w/ ENSO sensitivity (red bar when applicable). Next the script outputs a series of maps with complementary data csv(s) to accompany them. The first map shows the upper range of 2024-25 Outlook Predictions, the second map shows 2024-25 predicted increases in HTF since 2000, and the third shows the impact of ENSO on the predicted HTF annual outlook.

## ***HTF Annual Outlook Statistics***
This script HTF_annual_outlook_stats.py takes the results from the regression_fit.py and outputs two csv(s). The first is 2024_HTF_Annual_Outlook_station_stats.py which outputs different statistics on last years observations and the next year's prediction at each station. The second is 2024_HTF_Annual_Outlook_region.py which outputs summary statistics for each region [US (includes ALL stations), CONUS (continental US),  (Northeast), MID (Mid-Atlantic), SE (Southeast), EGC (Eastern Gulf Coast), WGC (Western Gulf Coast), SW (Southwest), NW (Northwest), AK (Alaska), CAR (Caribbean), and PAC (Pacific Islands)]

## ***AnnualHTFoutlook_Exploration_Notebook***
This is an interactive notebook to view results of the Annual Outlook both nationally and for individual stations. There are three required files which are included but can also be generated by the regression_fit.py. They are 2024_HTF_Annual_Outlook_full_data, HTF_Annual_Outlook_StationList_2024, and oni_annual_means_1950_2023.

#### For additional information, contact:

NOAA's Center for Operational Oceanographic Products and Services, [Coastal Hazards Branch](https://tidesandcurrents.noaa.gov/coastal_hazards.html)


nos.co-ops.chb@noaa.gov


## NOAA Open Source Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

## License

Software code created by U.S. Government employees is not subject to copyright in the United States (17 U.S.C. �105). The United States/Department of Commerce reserves all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.

