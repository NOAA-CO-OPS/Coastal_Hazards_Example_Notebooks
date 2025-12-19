# Coastal Hazards Example Notebooks

# Overview

This repository contains examples of Python codes in Jupyter Notebook/Google Colab with step-by-step explanations. These notebooks show users how to retrieve data from the NOAA's Center for Operational Oceanographic Products and Services' (CO-OPS') [Data API](https://api.tidesandcurrents.noaa.gov/api/prod/), [MetaData API](https://api.tidesandcurrents.noaa.gov/mdapi/prod/) and [Derived Products API](https://api.tidesandcurrents.noaa.gov/dpapi/prod/), visualize data, and do some level of statistical analysis. API documentations and data sources can be found at the top of each notebook. The example notebooks include:

## ***1. AnnualHTFoutlook_Exploration_Notebook***
The [Annual High Tide Flooding Outlook](https://tidesandcurrents.noaa.gov/high-tide-flooding/annual-outlook.html) provides the number of high tide flooding days predicted for the coming meteorological year (May to April). Data is supplemented with decadal projections for the year 2050, sea level rise scenarios, and high tide flood exposure maps to support long-term coastal planning. Summaries are provided for each region to account for geographical differences at the coast, and are accompanied by regional graphics to demonstrate potential high tide flooding impacts.

The ***Annual HTF Outlook Exploration Notebook*** is an interactive notebook to view results of the Annual Outlook both nationally and for individual stations. There are two required files which are included in the folder. They are 2025_HTF_Annual_Outlook_full_data and oni_annual_means_1950_2024.

## ***2. CO-OPS_12month_GEV_Extreme_Value_Analysis_Notebook***
NOAA's Center for Operational Oceanographic Products and Services (CO-OPS) provides critical information on coastal flood risk by analyzing the frequency of extreme high and low water levels. These statistics, often expressed as 1-year, 10-year, or 100-year return period elevations, are vital for coastal planning, infrastructure design, and resilience efforts.

The ***CO-OPS 12-Month GEV Extreme Value Analysis Notebook*** explores the statistical methods used to create the seasonal (12-month) probabilities found on the CO-OPS Extreme Water Levels website. The CO-OPS 12-Month GEV Extreme Value Analysis Notebook will walk users through preparing a monthly maxima/minima time series, running a Generalized Extreme Value (GEV) analysis on the data, and plotting the results as they would appear on the CO-OPS website.

## ***3. CO-OPS_API_Example_Notebook_HurricaneIan***
CO-OPS monitors water level and meteorological data during tropical cyclones and winter storms in real-time through the CO-OPS’ [Coastal Inundation Dashboard (CID)](https://tidesandcurrents.noaa.gov/inundationdb/). CID allows users to monitor elevated water level conditions along the coast when a tropical storm or hurricane watch/es or warning/s is/are issued. Also, CO-OPS provides Post-Event Peak Water Levels at NOS Stations through its [web mapping application](https://tidesandcurrents.noaa.gov/peakwaterlevels/index.html).

This notebook explores water levels and meteorological data observed during Hurricane Ian (2022) where CO-OPS’ stations along the coast captured significant [water levels at many locations](https://tidesandcurrents.noaa.gov/peakwaterlevels/index.html?year=2022&event=Hurricane%20Ian&datum=MHHW). These observations provide insight into the devastating impacts that Hurricane Ian had on the communities in its path and are critical for National Weather Service (NWS) hurricane specialists at the National Hurricane Center (NHC) who uses the data for storm surge forecast validation in real-time.

## ***4. CO-OPS_GEV_Extreme_Value_Analysis_Notebook***
NOAA's Center for Operational Oceanographic Products and Services (CO-OPS) provides critical information on coastal flood risk by analyzing the frequency of extreme high and low water levels. These statistics, often expressed as 1-year, 10-year, or 100-year return period elevations, are vital for coastal planning, infrastructure design, and resilience efforts.

The ***CO-OPS GEV Extreme Value Analysis Notebook*** explores the statistical methods used to create the annual exceedance probabilities (AEP) found on the CO-OPS Extreme Water Levels website. The CO-OPS GEV Extreme Value Analysis Notebook will walk users through detrending monthly maxima/minima time series, finding annual max/min water levels, running a Generalized Extreme Value (GEV) analysis on the data, and plotting the results as they would appear on the CO-OPS website.

## ***5. Extreme Water Levels: Station Exploration Notebook***

NOAA's Center for Operational Oceanographic Products and Services (CO-OPS) is the nation's source for coastal inundation data, providing vital statistics on the frequency and magnitude of extreme water levels. This information is critical for understanding future flood risk and is featured in products like the CO-OPS [Extreme Water Levels website](https://tidesandcurrents.noaa.gov/est/) and the [2022 Sea Level Rise Technical Report](https://oceanservice.noaa.gov/hazards/sealevelrise/sealevelrise-tech-report.html).

The ***Extreme Water Levels: Station Exploration Notebook*** allows users to review, plot, and examine the data and methods behind these products. It walks through how different extreme value analyses (like GEV and GPD) are used to calculate risk, explore past extreme events, and visualize how the probability of extreme water levels may change in the future.

*Note that, additional required data (`.csv` files) is included in the sub-directory.*

## ***6. HTF_APIretrieval_MultiYear_Heatmap***
The ***HTF API retrieval MultiYear Heatmap*** notebook is designed to show users the heatmap of historical flood hours/days, where the observed water level was more than the specified [National Ocean Service (NOS) flooding thresholds](https://www.tidesandcurrents.noaa.gov/publications/techrpt86_PaP_of_HTFlooding.pdf).

Users can explore CO-OPS' available oceanographic and meteorological data products at https://tidesandcurrents.noaa.gov.

## ***7. HTF_Outlook_APIretrieval_plotting***
Above-normal tides can trigger high tide flooding, disrupting coastal communities. This flooding can occur on sunny days and in the absence of storms. More severe flooding may occur if high tides coincide with heavy rains, strong winds, or large waves. As sea levels continue to rise, our coastal communities will experience more frequent high tide flooding - a National average of 45 to 85 days per year by 2050. Predicting the frequency of high tide flooding in the future helps coastal communities plan for and mitigate flooding impacts.

The [Annual High Tide Flooding Outlook](https://tidesandcurrents.noaa.gov/high-tide-flooding/annual-outlook.html) provides the number of high tide flooding days predicted for the coming meteorological year (May to April). Data is supplemented with decadal projections for the year 2050, sea level rise scenarios, and high tide flood exposure maps to support long-term coastal planning. Summaries are provided for each region to account for geographical differences at the coast, and are accompanied by regional graphics to demonstrate potential high tide flooding impacts.

The ***HTF Outlook API retrieval plotting*** notebook provides a glimpse of what CO-OPS does to deliver the Annual High Tide Flooding Outlook. Users will have the opportunity to explore a single station and visualize data. 

## ***8. Sea_Level_Rise_Station_Exploration_Notebook***
CO-OPS is the nation's authoritative source for coastal inundation data and sea level trends through its network of long-term water level gauges, the National Water Level Observation Network (NWLON). Using these products, communities can plan and implement long-term adaptation plans to protect their economy from coastal hazards.

The ***Sea Level Rise Station Exploration Notebook*** will review, plot, and perform statistical analysis on past local sea level data and explore future trajectories. This notebook will walk through how sea level trends are calculated and how different graphics that are part of CO-OPS [Sea Level Trends products](https://tidesandcurrents.noaa.gov/sltrends/sltrends.html) are produced.

*Note that, additional required data is included in the sub-directory*

## ***9. SLR_Scenarios_APIretrieval_plotting***
The projection of future sea levels that are shown in CO-OPS' [Relative Sea Level Trends](https://tidesandcurrents.noaa.gov/sltrends/sltrends.html) page were released in 2022 by the [U.S. interagency task force](https://oceanservice.noaa.gov/hazards/sealevelrise/sealevelrise-tech-report-sections.html) in preparation for the Fifth National Climate Assessment. The projections for 5 sea level change scenarios are expected to assist decision makers in responding to local relative sea level rise. 

The ***SLR Scenarios API retrieval plotting*** notebook is designed to provide users the flexibility through building api to generate queries from CO-OPS' Data API, MetaData API, and Derived Product API and plot the SLR scenarios. 

## ***10. Sea Level Trend Calculation Notebook***

NOAA's Center for Operational Oceanographic Products and Services (CO-OPS) is the nation's source for [coastal inundation data and sea level trends](https://tidesandcurrents.noaa.gov/sea_level_info.html) through its network of long-term water level gauges. These gauges provide the foundational data for CO-OPS's official [Sea Level Trends](https://tidesandcurrents.noaa.gov/sltrends/sltrends.html) products, which are critical for communities to plan and adapt to long-term coastal change.

The ***Sea Level Trend Calculation Notebook*** walks users through the exact methodology CO-OPS uses to calculate these trends. It allows users to explore historical sea level data and replicate the calculations for both a single Period of Record (POR) trend and a split trend (breakpoint at 1993).

#### For additional information, contact:
NOAA's [Center for Operational Oceanographic Products and Services](https://tidesandcurrents.noaa.gov/)
User Services: tide.predictions@noaa.gov


## NOAA Open Source Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

## License

Software code created by U.S. Government employees is not subject to copyright in the United States (17 U.S.C. �105). The United States/Department of Commerce reserve all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.
