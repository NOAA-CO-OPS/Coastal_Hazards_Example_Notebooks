# Coastal Hazards Example Notebooks

# Overview

This repository contains examples of Python code in Jupyter Notebooks/Gooogle Colabs with step-by-step explanations. These notebooks show users how to retrieve data from the NOAA's Center for Operational Oceanographic Products and Services' (CO-OPS') Data, MetaData and Drived Products APIs, visualize data, and do some level statistical analysis. API documentations and data sources can be found at the top of each notebook. The example notebooks include:

### Sea_Level_Rise_Station_Exploration_Notebook
CO-OPS is the nation's authoritative source for coastal inundation data and sea level trends through its network of long-term water level gauges, the National Water Level Observation Network (NWLON). Using these trends as a guide, communities can implement long-term adaptation plans to protect their economy from coastal hazards.

In this notebook, you will review, plot, and perform statisical analysis on past local sea level data and explore future trajectories. This notebook will walk through how sea level trends are calculated and how different graphics that are part of CO-OPS Sea Level Trends products (https://tidesandcurrents.noaa.gov/sltrends/sltrends.html) are produced.

### CO-OPS_API_Example_Notebook_HurricaneIan.ipynb
CO-OPS monitors water level and meteorological data during tropical cyclones and winter storms in real-time through the CO-OPS’ Coastal Inundation Dashboard (CID) (https://tidesandcurrents.noaa.gov/inundationdb/) web mapping application. CID allows users to monitor elevated water level conditions along the coast when a tropical storm or hurricane watch/es or warning/s is/are issued. Also, CO-OPS provides Post-Event Peak Water Levels at NOS Stations through its web mapping application (https://tidesandcurrents.noaa.gov/peakwaterlevels/index.html).

This notebook explores water levels and meteorological data observed during Hurricane Ian (2022) where CO-OPS’ stations along the coast captured significant water levels at many locations (https://tidesandcurrents.noaa.gov/peakwaterlevels/index.html?year=2022&event=Hurricane%20Ian&datum=MHHW). These observations provide insight into the devastating impacts that Hurricane Ian had on the communities in its path and are critical for National Weather Service (NWS) hurricane specialists at the National Hurricane Center (NHC) who uses the data for storm surge forecast validation in real-time.

### HTF_Outlook_APIretrieval_plotting.ipynb
Above-normal tides can trigger high tide flooding, disrupting coastal communities. This flooding can occur on sunny days and in the absence of storms. More severe flooding may occur if high tides coincide with heavy rains, strong winds, or large waves. As sea levels continue to rise, our coastal communities will experience more frequent high tide flooding - a National average of 45 to 85 days per year by 2050. Predicting the frequency of high tide flooding in the future helps coastal communities plan for and mitigate flooding impacts.

The Annual High Tide Flooding Outlook (https://tidesandcurrents.noaa.gov/high-tide-flooding/annual-outlook.html) provides the number of high tide flooding days predicted for the coming meteorological year (May to April). Data is supplemented with decadal projections for the year 2050, sea level rise scenarios, and high tide flood exposure maps to support long-term coastal planning. Summaries are provided for each region to account for geographical differences at the coast, and are accompanied by regional graphics to demonstrate potential high tide flooding impacts.

This notebook provides you a glimpse of what CO-OPS does to provide the Annual High Tide Flooding Outlook. Users will have teh opportunity to explore a single station and visualize data. 

### SLR_Scenarios_APIretrieval_plotting.ipynb
The projection of future sea levels that are shown in CO-OPS' Relative Sea Level Trends (https://tidesandcurrents.noaa.gov/sltrends/sltrends.html) page were released in 2022 by a U.S. interagency task force (https://oceanservice.noaa.gov/hazards/sealevelrise/sealevelrise-tech-report-sections.html) in preparation for the Fifth National Climate Assessment. The projections for 5 sea level change scenarios are expected to assist decision makers in responding to local relative sea level rise. 

This notebook is designed to give you more flexiblity through building api to generate queries from CO-OPS' Data API, MetaData API, and Derived Product API.

### HTF_APIretrieval_MultiYear_Heatmap.ipynb

This notebook is designed to show you the heatmap of historical flood hours/days, where the observed water level was more than specified National Ocean Service (NOS) flooding thresholds (https://www.tidesandcurrents.noaa.gov/publications/techrpt86_PaP_of_HTFlooding.pdf).

Users can explore CO-OPS' available oceanographic and meteorological data products at https://tidesandcurrents.noaa.gov.

#### For additional information, contact:
NOAA Center for Operational Oceanographic Products and Services,
Tigist jima, Lindsay Abrams, Karen Kavanaugh, and/or Bailey Armos
tigist.jima@noaa.gov, lindsay.abrams@noaa.gov, karen.kavanaugh@noaa.gov and/or 
bailey.armos@noaa.gov

## NOAA Open Source Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

## License

Software code created by U.S. Government employees is not subject to copyright in the United States (17 U.S.C. �105). The United States/Department of Commerce reserve all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.
