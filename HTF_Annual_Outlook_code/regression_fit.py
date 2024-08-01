import pandas as pd
import datetime as dt
import configparser
import numpy as np
import statsmodels.formula.api as smf
import sklearn.metrics as skl
import statsmodels.tools as smt

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
start_year = int(data_params['start_year'])
run_year = int(data_params['run_year'])
pval_lim = float(data_params['pvalue_limit'])

today = dt.datetime.now()
run_date = today.strftime("%m/%d/%Y")


stn_df = pd.read_csv(stn_list)

# Creates a dataframe using the NOAA IDs for all stations on the station list specified in the config file

stn_list_df = pd.DataFrame({'stnId':stn_df['NOAA_ID']})

# Obtain forecast average from the value obtained by enso_forecast_avg.py

avg_file = pd.read_csv(f'{data_dir}\\average_all_models.txt',header=None)

forecast_avg = avg_file.iloc[0][0]

# Obtain Annual Averages obtained by enso_oni_download.py

annual_means = pd.read_csv(f'{data_dir}\\annual_means.csv')

# Obtain historic flood counts obtained by historic_flood_counts.py

historic_flood_cts_all = pd.read_csv(f'{data_dir}\\met_historic_flood_cts.csv')

historic_flood_cts = historic_flood_cts_all[historic_flood_cts_all['metYear'].between(start_year,run_year-1)]

historic_flood_cts = historic_flood_cts.dropna(subset=['minCount'])

flood_cts_group = historic_flood_cts.groupby('stnId')

oni = pd.concat([annual_means['Year'],annual_means['ONI Met Year']], axis=1, keys = ['metYear','oni'])

flood_cts_oni = pd.merge(historic_flood_cts,oni,on='metYear')

# Filter to station list
flood_cts_oni = pd.merge(flood_cts_oni,stn_list_df,on=['stnId'])

# Creates an empty dataframe with the columns for the full results

full_results_df = pd.DataFrame(columns=['stnID','stnName','lowConf','highConf','projectDate','method',\
                                    'projectMethodID','projectMethod',\
                                   'quad_enso_fit','quad_enso_coef_pval',\
                                    'quad_enso_pval','quad_enso_pred','quad_enso_rmse',\
                                    'quad_fit','quad_coef_pval',\
                                    'quad_pval','quad_pred','quad_rmse',\
                                    'lin_enso_fit','lin_enso_coef_pval',\
                                    'lin_enso_pval','lin_enso_pred','lin_enso_rmse',\
                                    'lin_fit','lin_coef_pval',\
                                    'lin_pval','lin_pred','lin_rmse',\
                                    'no_t_quad_fit','no_t_quad_coef_pval',\
                                    'no_t_quad_pval','no_t_quad_pred','no_t_quad_rmse',\
                                    'no_t_lin_fit','no_t_lin_coef_pval',\
                                    'no_t_lin_pval','no_t_lin_pred','no_t_lin_rmse','avg19','stdev19','pred_2k'])


for station, station_df in flood_cts_oni.groupby('stnId'):

    # Define metYear as x, minCount as y, and ONI as z
    format_df = pd.DataFrame({'x':station_df['metYear'],'y':station_df['minCount'],'z':station_df['oni']})

    test_df = pd.DataFrame({'x':[run_year],'z':[forecast_avg]})

    # Create dataframe with the year 2000 values

    test_2k = pd.DataFrame({'x':[2000]})

    # Linear 
    # y = ax + b

    #Create model
    lin_model = smf.ols(formula='y ~ x',data=format_df).fit(method='qr')
    
    # Define variable for the coefficients
    lin_params = lin_model.params
    
    # Create a predicted value for run year
    lin_result = lin_model.predict(test_df)[0]

    # Create a string of the formula 

    lin_formula = f"{str(lin_params['x'])}*x + {str(lin_params['Intercept'])}"

    lin_pred = lin_model.predict(format_df)
  
    # Calculate the rmse which will be used for high/low predictions
    
    lin_rmse = np.sqrt(skl.mean_squared_error(format_df['y'],lin_pred))
    
    # Round pvalues to 3 decimal points
    lin_model.pvalues = lin_model.pvalues.round(3)

    # Checking the count of coefficients that are non-zero. The intercept is the first element of the params series so it is skipped

    lin_nz_ct = 0

    for param in lin_params[1:]:
        if(param!=0.0):
            lin_nz_ct+=1

    # Prediction for 2000, to be used in graphs

    lin_2k = lin_model.predict(test_2k)[0]

    # Linear w/ ENSO
    # y = ax + bz + c

    lin_model_enso = smf.ols(formula='y ~ x + z',data=format_df).fit(method='qr')

    lin_enso_params = lin_model_enso.params
 
    lin_enso_result = lin_model_enso.predict(test_df)[0]

    lin_enso_formula = f"{str(lin_enso_params['x'])}*x + {str(lin_enso_params['z'])}*z + {str(lin_enso_params['Intercept'])}"
    
    lin_pred_enso = lin_model_enso.predict(format_df)

    lin_rmse_enso = np.sqrt(skl.mean_squared_error(format_df['y'],lin_pred_enso))

    lin_model_enso.pvalues = lin_model_enso.pvalues.round(3)

    lin_enso_nz_ct = 0

    for param in lin_enso_params[1:]:
        if(param!=0.0):
            lin_enso_nz_ct+=1

    # Quadratic 
    # y =a*x^2 + b*x + c

    at^2+btONI+ct+dONI^2+eONI+f
    
    quad_model = smf.ols(formula='y ~ np.power(x,2)+x',data=format_df).fit(method='qr')

    quad_params = quad_model.params

    quad_result = quad_model.predict(test_df)[0]

    quad_formula = f"{str(quad_params['np.power(x, 2)'])}*x^2 + {str(quad_params['x'])}*x + {str(quad_params['Intercept'])}"

    quad_pred = quad_model.predict(format_df)

    quad_rmse = np.sqrt(skl.mean_squared_error(format_df['y'],quad_pred))

    quad_model.pvalues = quad_model.pvalues.round(3)

    quad_2k = quad_model.predict(test_2k)[0]

    quad_nz_ct = 0

    for param in quad_params[1:]:
        if(param!=0.0):
            quad_nz_ct+=1


    # Quad w/ ENSO
    # y = (a * x**2) + (b * x * z) + (c*x) + (d*z**2) + e*z + f

    quad_model_enso = smf.ols(formula='y ~ np.power(x,2) + x*z + x + np.power(z,2) + z',data=format_df).fit(method='qr')

    quad_enso_params = quad_model_enso.params

    quad_enso_result = quad_model_enso.predict(test_df)[0]

    quad_enso_formula = f"{quad_enso_params['np.power(x, 2)']}*x^2 + {str(quad_enso_params['x:z'])}*x*z + {str(quad_enso_params['x'])}*x + {str(quad_enso_params['np.power(z, 2)'])}*z^2 + {str(quad_enso_params['z'])}*z + {str(quad_enso_params['Intercept'])}"

    quad_enso_pred = quad_model_enso.predict(format_df)

    quad_rmse_enso = np.sqrt(skl.mean_squared_error(format_df['y'],quad_enso_pred))

    quad_model_enso.pvalues = quad_model_enso.pvalues.round(3)

    quad_enso_nz_ct = 0

    for param in quad_enso_params[1:]:
        if(param!=0.0):
            quad_enso_nz_ct+=1

    # No Trend w/ ENSO Linear
    # (a*z) + b

    no_trend_lin_model_enso = smf.ols(formula='y ~ z',data=format_df).fit(method='qr')

    no_trend_lin_params = no_trend_lin_model_enso.params

    no_trend_lin_result = no_trend_lin_model_enso.predict(test_df)[0]

    no_trend_lin_formula = f"{str(no_trend_lin_params['z'])}*z + {str(no_trend_lin_params['Intercept'])}"

    no_trend_lin_pred = no_trend_lin_model_enso.predict(format_df)

    no_trend_lin_rmse = np.sqrt(skl.mean_squared_error(format_df['y'],no_trend_lin_pred))

    no_trend_lin_model_enso.pvalues = no_trend_lin_model_enso.pvalues.round(3)
   
    no_t_lin_nz_ct = 0

    for param in no_trend_lin_params[1:]:
        if(param!=0.0):
            no_t_lin_nz_ct+=1


    # No Trend w/ ENSO Quadratic
    # (a*z**2) + (b*z) + c

    no_trend_quad_model_enso = smf.ols(formula='y ~ np.power(z,2) + z',data=format_df).fit(method='qr')

    no_trend_quad_params = no_trend_quad_model_enso.params

    no_trend_quad_result = no_trend_quad_model_enso.predict(test_df)[0]

    no_trend_quad_formula = f"{str(no_trend_quad_params['np.power(z, 2)'])}*z^2 + {str(no_trend_quad_params['z'])}*z + {str(no_trend_quad_params['Intercept'])}"
    
    no_trend_quad_pred = no_trend_quad_model_enso.predict(format_df)

    no_trend_quad_rmse = np.sqrt(skl.mean_squared_error(format_df['y'],no_trend_quad_pred))

    no_trend_quad_model_enso.pvalues = no_trend_quad_model_enso.pvalues.round(3)
    
    no_t_quad_nz_ct = 0

    for param in no_trend_quad_params[1:]:
        if(param!=0.0):
            no_t_quad_nz_ct+=1

    nineteen_df = format_df.copy()
    nineteen_df['x'] = nineteen_df['x'].astype(int)
    nineteen_df = nineteen_df[nineteen_df['x']>=int(run_year)-19]
    mean_nineteen = np.mean(nineteen_df['y'])
    std_nineteen = nineteen_df['y'].std() 


    # Selecting the regression method that will be used for the outlook based on a decision tree, and calculates prediction and RMSE

    if(quad_model_enso.pvalues['np.power(x, 2)'] < pval_lim and quad_enso_params['np.power(x, 2)'] > 0.0 and (quad_model_enso.pvalues['z'] < pval_lim or quad_model_enso.pvalues['x:z'] < pval_lim or quad_model_enso.pvalues['np.power(z, 2)'] < pval_lim) and quad_enso_nz_ct > 1):
        method = "Quad ENSO"
        proj_id = 4
        proj_method = "Quadratic With ENSO Sensitivity"
        result = quad_enso_result
        rmse = quad_rmse_enso
        formula = quad_enso_params.to_dict()
        trend_formula = quad_params.to_dict()
        pred_2k = quad_2k
    elif(quad_model.pvalues['np.power(x, 2)'] < pval_lim and quad_params['np.power(x, 2)'] > 0.0 and quad_nz_ct > 1):
        method = "Quad"
        proj_id = 3
        proj_method = "Quadratic"
        result = quad_result
        rmse = quad_rmse
        formula = quad_params.to_dict()
        trend_formula = np.nan
        pred_2k = quad_2k
    elif(lin_model_enso.pvalues['x'] < pval_lim and lin_model_enso.pvalues['z'] < pval_lim and lin_enso_nz_ct > 1 ):
        method = "Lin ENSO"
        proj_id = 2
        proj_method = 'Linear With ENSO Sensitivity'
        result = lin_enso_result
        rmse = lin_rmse_enso
        formula = lin_enso_params.to_dict()
        trend_formula = lin_params.to_dict()
        pred_2k = lin_2k
    elif(lin_model.pvalues['x'] < pval_lim and lin_nz_ct > 0):
        method = "Lin"
        proj_id = 1
        proj_method = 'Linear'
        result = lin_result
        rmse = lin_rmse
        formula = lin_params.to_dict()
        trend_formula = np.nan
        pred_2k = lin_2k
    elif(no_trend_quad_model_enso.pvalues['np.power(z, 2)'] < pval_lim and no_t_quad_nz_ct > 1):
        method = "No Trend Quad"
        proj_id = 5
        proj_method = 'No Temporal Trend with Quadratic ENSO Sensitivity'
        result = no_trend_quad_result
        rmse = no_trend_quad_rmse
        formula = no_trend_quad_params.to_dict()
        trend_formula = np.nan
        pred_2k = mean_nineteen
    elif(no_trend_lin_model_enso.pvalues['z'] < pval_lim and no_t_lin_nz_ct > 0):
        method = "No Trend Linear"
        proj_id = 5
        proj_method = 'No Temporal Trend with Linear ENSO Sensitivity'
        result = no_trend_lin_result
        rmse = no_trend_lin_rmse
        formula = no_trend_lin_params.to_dict()
        trend_formula = np.nan
        pred_2k = mean_nineteen
    else:
        method = "19 Yr Avg"
        proj_id = 6
        proj_method = 'No Temporal Trend with 19 year average'
        result = mean_nineteen
        rmse = std_nineteen # StDev is used to calc upper/lower instead of RMSE but use same variable
        formula = np.nan
        trend_formula = np.nan
        pred_2k = mean_nineteen

    # Calculates upper and lower bounds for predictions, using the model prediction +/- RMSE

    upper = round(result + rmse)
    lower = round(result - rmse)
 
    # If lower bound would be less than 0, set as 0

    if lower < 0:
        lower = 0

    # Creates a temporary dataframe to store individual station results

    temp_df = pd.DataFrame({'stnID':[station],'stnName':np.nan,'lowConf':[lower],'highConf':[upper],'projectDate':[run_date],\
                            'projectMethodID':[proj_id], 'projectMethod': [proj_method], 'method':[method],\
                            'quad_enso_fit':[quad_enso_params.to_dict()],'quad_enso_coef_pval':[quad_model_enso.pvalues.to_dict()],\
                            'quad_enso_pval':[quad_model_enso.f_pvalue],\
                            'quad_enso_pred':[quad_enso_result],'quad_enso_rmse':[quad_rmse_enso],\
                            'quad_fit':[quad_params.to_dict()],'quad_coef_pval':[quad_model.pvalues.to_dict()],\
                            'quad_pval':[quad_model.f_pvalue],\
                            'quad_pred':[quad_result],'quad_rmse':[quad_rmse],\
                            'lin_enso_fit':[lin_enso_params.to_dict()],'lin_enso_coef_pval':[lin_model_enso.pvalues.to_dict()],\
                            'lin_enso_pval':[lin_model_enso.f_pvalue],\
                            'lin_enso_pred':[lin_enso_result],'lin_enso_rmse':[lin_rmse_enso],\
                            'lin_fit':[lin_params.to_dict()],'lin_coef_pval':[lin_model.pvalues.to_dict()],\
                            'lin_pval':[lin_model.f_pvalue],\
                            'lin_pred':[lin_result],'lin_rmse':[lin_rmse],\
                            'no_t_quad_fit':[no_trend_quad_params.to_dict()],'no_t_quad_coef_pval':[no_trend_quad_model_enso.pvalues.to_dict()],\
                            'no_t_quad_pval':[no_trend_quad_model_enso.f_pvalue],\
                            'no_t_quad_pred':[no_trend_quad_result],'no_t_quad_rmse':[no_trend_quad_rmse],\
                            'no_t_lin_fit':[no_trend_lin_params.to_dict()],'no_t_lin_coef_pval':[no_trend_lin_model_enso.pvalues.to_dict()],\
                            'no_t_lin_pval':[no_trend_lin_model_enso.f_pvalue],\
                            'no_t_lin_pred':[no_trend_lin_result],'no_t_lin_rmse':[no_trend_lin_rmse],
                            'avg19':[mean_nineteen],'stdev19':[std_nineteen],'pred_2k':[pred_2k]})
    
    full_results_df = pd.concat([full_results_df,temp_df],ignore_index=True)

    temp_df_2 = pd.DataFrame({'stnID':[station],'stnName':np.nan,'formula':[formula],'trend':[trend_formula]})
    
full_results_df['stnName'] = stn_df['Name']

# Saves the full results

full_results_df.to_csv(f'{data_dir}\\{run_year}_HTF_Annual_Outlook_full_data.csv',index=False)

# Saves the main outlook file 

outlook_df = full_results_df[['stnID','stnName','lowConf','highConf','projectDate',\
                                    'projectMethodID','projectMethod','method']]

outlook_df.to_csv(f'{data_dir}\\{run_year}_HTF_Annual_Outlook.csv',index=False)