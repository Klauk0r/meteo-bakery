###################################################################################################################
# DEPENDENCIES

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

from lightgbm import LGBMRegressor
import shap


###################################################################################################################
# FEATURE ENGINEERING

###################################################################################################################
# function for extracting datetime information
def get_time_info(df, date_col, parse_to_datetime=False):
    """Takes a pandas dataframe and extracts datetime information from datetime column.

    Args:
        df (pd.DataFrame): A pandas dataframe
        date_col (str): A datetime column used for extracting datetime information
        parse_to_datetime (bool, optional): Parse date column if not yet datetime object: Defaults to False

    Returns:
        pd.DataFrame: A pandas dataframe containing datetime information
    """
    # parse to datetime if necessary
    if parse_to_datetime==True:
        df[date_col] = pd.to_datetime(df[date_col])
    
    #df['date'] = df[date_col].dt.date
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week'] = df[date_col].dt.isocalendar().week
    df['day_of_month'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    return df

###################################################################################################################
# function for generating sine- and cosine-transformed month
def transform_month(df, var_name):
    """Takes in month and generates sine- and cosine-transformed components

    Args:
        df (pd.DataFrame): A pandas dataframe
        var_name (str): Name of variable that encodes for the month

    Returns:
        pd.DataFrame: A pandas dataframe with columns on sine- and cosine-transformed month
    """
    df['month_sin'] = df[var_name].apply(lambda x: np.sin(np.array(x) * np.pi /6))
    df['month_cos'] = df[var_name].apply(lambda x: np.cos(np.array(x) * np.pi /6))
    
    return df

###################################################################################################################
# function for generating x and y components of wind direction
def get_wind_components(df, var_name):
    """Converts wind direction in degrees into x and y components. Returns a pandas dataframe with added wind components.

    Args:
        df (pd.DataFrame): A pandas dataframe
        var_name (str): Name of the variable that measures wind direction [in degrees]

    Returns:
        pd.DataFrame: A pandas dataframe with added wind components
    """
    df['wind_dir_x'] = df[var_name].apply(lambda x: np.cos(np.array(x) * np.pi /180))
    df['wind_dir_y'] = df[var_name].apply(lambda x: np.sin(np.array(x) * np.pi /180))
    
    return df

###################################################################################################################
# utility function for extracting summary statistics from daily weather recordings
def extract_daily_statistics(df, date_col, columns, daytimes=False):
    """Extracts daily summary statistics of hourly weather data. 
        Returns a dataframe with mean, min, max and std values for weather recordings between 06:00 - 20:00 
        and also returns mean values for three daily time period, i.e. 06:00-10:00, 11:00-15:00, and 16:00-20:00.
        ATTENTION: This function requires the dataframe to contain a datetime column with hourly information!

    Args:
        df (pd.DataFrame): Dataframe containing ourly weather recordings
        date_col (str): datetime column used to extract time frames
        columns (list): A list of columns containing weather variable names contained in the input dataframe.

    Returns:
        pd.DataFrame: Dataframe containing daily summary statistics of weather data
    """
    
    # parse to datetime and create date column in input dataframe to be used for grouping later
    df[date_col] = pd.to_datetime(df[date_col],utc=True)
    df['date'] = df[date_col].dt.date

    # intialize dataframe for summary statistics
    summary_stats = pd.DataFrame({'date': df[date_col].dt.date.unique()})
    summary_stats.set_index('date', inplace=True)
    
    # subselect dataframe for calculating whole-day summary statistics
    df_day = df[df[date_col].dt.hour.between(6, 20)]

    if daytimes:
        # subselect dataframe for specified time frames
        # 06:00-10:00
        df_6_10 = df[df[date_col].dt.hour.between(6, 10)]
        # 11:00-15:00
        df_11_15= df[df[date_col].dt.hour.between(11, 15)]
        # 16:00-20:00
        df_16_20= df[df[date_col].dt.hour.between(16, 20)]

    for col in columns:
        # calculate different summary statistics over complete daytime period
        summary_stats[[col+'_mean', col+'_min', col+'_max', col+'_std']] = df_day.groupby('date')[col].agg(['mean', 'min', 'max', 'std'])
        
        if daytimes:
            # calculate mean values for specified time frames
            # 06:00-10:00
            summary_stats[col+'_06_10'] = df_6_10.groupby('date')[col].agg(['mean'])
            # 11:00-15:00
            summary_stats[col+'_11_15'] = df_11_15.groupby('date')[col].agg(['mean'])
            # 16:00-20:00
            summary_stats[col+'_16_20'] = df_16_20.groupby('date')[col].agg(['mean'])

    return summary_stats

###################################################################################################################
# function for extracting seasonal deviations from daily weather data
def get_deviations(df, period=365, stat='_mean'):
    '''Gets weather residuals from seasonal and trend influences. Requires a dataframe of weather data aggregated on day-level.

    Args:
        df (pd.DataFrame): A TimeSeries as DataFrame with time as an index and target value(s) as columns. If 2d, individual series are in columns. Must contain 2 complete cycles.
        period (int, optional): Period of the series. Defaults to 365 (one year).
        stat (str, optional): Summary statistic to extract. Can be '_mean', '_min', '_max', '_std', '_06_10', '_11_15' or '_16_20'. Defaults to '_mean'.

    Returns:
        Pandas DataFrame: contains the residuals for each of the columns of the provided DataFrame
    '''
    elms = [elm for elm in df.columns if stat in elm]
    dev_df = pd.DataFrame()
    for e in elms:
        resdf = pd.DataFrame(
                        seasonal_decompose(x=df[f'{e}'].fillna(method='ffill'),
                                            model='additive', 
                                            two_sided=False,
                                            period=period
                                            )._resid
                            )
        resdf.rename(columns={'resid':f'{e}_dev'}, inplace=True)
        dev_df = pd.concat([dev_df, resdf],axis=1,join='outer')
    return dev_df

###################################################################################################################
# function for calculating day-to-day changes in weather data
def get_weather_changes(df, columns, comp=1):
    '''calculates changes for columns in TimeSeries dataframe containing daily weather data.

    Args:
        df (pd.DataFrame): A DataFrame. 
        columns (iterable): iterable column names of df. 
        comp (int, optional): shift which to compare to. Defaults to 1.

    Returns:
        DataFrame: Table of the changes of each category compared to the index comp steps before.
    '''
    change_df = pd.DataFrame()
    for c in columns:
        cdf=(df[c]-(df[c].shift(comp))).to_frame(name=f'{c}_change')
        change_df = pd.concat([change_df, cdf],axis=1,join='outer')
    return change_df

###################################################################################################################
# function for generating lagged features
def get_lag_features(df, grouping_vars, feature, lags):
    """Takes in a stacked time series dataframe and generates lag features for defined lags and returns dataframe with lags as
    additional columns.

    Args:
        df (pd.DataFrame): Stacked time series dataframe
        grouping_vars (list): A list of grouping variables. Currently accepts only a list of two variables.
        feature (str): Name of the feature, for which lags should be generated.
        lags (list): A list of lags to generate lag features

    Returns:
        pd.DataFrame: A dataframe containing the lag features as additional columns.
    """
    # initialize empty dataframe
    df_lag = pd.DataFrame({})
    
    for i, group in enumerate(product(df[grouping_vars[0]].unique(), df[grouping_vars[1]].unique())):
        # subselect time series and generate lag features
        ts = df[(df[grouping_vars[0]]==group[0]) & (df[grouping_vars[1]]==group[1])].copy()
        # map feature to dictionary
        target_map = ts[feature].to_dict()
        # iterate over every lag, map feature according to lag and append to dataframe
        for lag in lags:
            ts[f'{feature}_lag_{lag}'] = (ts.index - pd.Timedelta(f'{lag} days')).map(target_map)
            
        df_lag = pd.concat([df_lag, ts], axis=0)
    
    return df_lag

###################################################################################################################
# function for generating lead features
def get_lead_features(df, grouping_vars, feature, leads):
    """Takes in a stacked time series dataframe and generates lead features for defined leads and returns dataframe with leads as
    additional columns.

    Args:
        df (pd.DataFrame): Stacked time series dataframe
        grouping_vars (list): A list of grouping variables. Currently accepts only a list of two variables.
        feature (str): Name of the feature, for which leads should be generated.
        leads (list): A list of leads to generate lead features

    Returns:
        pd.DataFrame: A dataframe containing the lead features as additional columns.
    """
    # initialize empty dataframe
    df_lead = pd.DataFrame({})
    
    for i, group in enumerate(product(df[grouping_vars[0]].unique(), df[grouping_vars[1]].unique())):
        # subselect time series and generate lead features
        ts = df[(df[grouping_vars[0]]==group[0]) & (df[grouping_vars[1]]==group[1])].copy()
        # map feature to dictionary
        target_map = ts[feature].to_dict()
        # iterate over every lead, map feature according to lead and append to dataframe
        for lead in leads:
            ts[f'{feature}_lead_{lead}'] = (ts.index + pd.Timedelta(f'{lead} days')).map(target_map)
            
        df_lead = pd.concat([df_lead, ts], axis=0)
    
    return df_lead


###################################################################################################################
# MODELING

###################################################################################################################
# cross-validation for naive seasonal baseline corrected for holiday effects
def crossval_naive(df_train, grouping_vars, target, splits=52, test_size=7, gap=0):
    """Cross-Validation for Naive Seasonal Baseline:
    Takes in a training dataset of stacked time series and performs TimeSeriesSplit Cross-Validation for Naive Seasonal 
    baseline model for each of those time series. Returns a dataframe of cross-validation results containing mean MAPE scores 
    and corresponding standard deviations from cross-validation of each individual time series.
    
    ATTENTION: This function corrects the prediction and validation sets for possible NaNs due to branch closures on holidays. However, this correction
    assumes such NaNs to be represented as 1. Thus, any NaNs due to branch closures on holidays should be converted into 1 before applying this function.
    
    Further Explanation:
    The Naive baseline model uses a drift of 7 days, i.e. it takes the sales from the preceding 7 days as a prediction for the next 7 days.
    If an NaN due to branch closure on holidays is contained in the predictions, the NaN is replaced by the sales data one week earlier
    (i.e. the sales 14 days ago are used as a prediction in such a case). If any NaN is contained in the validation set due to holiday closure,
    the datapoint is deleted and only a prediction for a reduced validation set is made. 

    Args:
        df_train (pd.DataFrame): A training dataframe containing stacked time series data
        grouping_vars (list): A list of grouping variables, according to which training data is stacked. Currently accepts only a list of two variables.
        target (str): Prediction target
        splits (int, optional): Number of splits for Cross-Validation. Defaults to 52 (1 fold / week).
        test_size (int, optional): Size of validation set (i.e. forecasting horizon). Defaults to 7 days.
        gap (int, optional): Time gap between end of training and start of validation set. Defaults to 0.

    Returns:
        pd.DataFrame: A dataframe with cross-validation results, containing mean MAPE scores and respective standard deviations for each time series
    """
    # initialize dataframe for evaluation scores
    cval = pd.DataFrame({'group': [], 'MAPE_mean': [], 'MAPE_std': []})

    # iterate over all individual series and perform cross-validation
    for i, group in enumerate(product(df_train[grouping_vars[0]].unique(), df_train[grouping_vars[1]].unique())):

        # subselect time series
        ts = df_train[(df_train[grouping_vars[0]]==group[0]) & (df_train[grouping_vars[1]]==group[1])].copy()
        
        # perform cross validation
        tss = TimeSeriesSplit(n_splits=splits, test_size=test_size, gap=gap)
        # initialize scores list for append MAPE scores from individual folds and start cross-validation
        scores = []
        for train_i, val_i in tss.split(ts):

            y_train = ts.iloc[train_i][target]
            y_val = ts.iloc[val_i][target]
            
            # correct for holiday effects in predicted values based on training set if necessary
            # if 1 (representing missing values on a holiday) is in predicted y-values, replace by sales 14 days ago
            if 1 in y_train[-7:].unique():
                idx_train = [i for i in range(len(y_train[-7:].tolist())) if y_train[-7:].tolist()[i]==1]
                idx_train = [i-7 for i in idx_train]
                idx_train_lag = [i-7 for i in idx_train]
                y_train_repl = y_train.copy()
                y_train_repl.iloc[idx_train] = [x for x in y_train.iloc[idx_train_lag]]
                y_pred = y_train_repl[-7:]
            else:
                y_pred = y_train[-7:]

            # correct for holiday effects in validation set if necessary
            # if 1 (representing missing values on a holiday) is in validation set, drop elements at corresponding index position in both y_val and y_pred
            if 1 in y_val.unique():
                idx_val = [i for i in range(len(y_val.tolist())) if y_val.tolist()[i]==1]
                y_val = y_val.drop(y_val.index[idx_val])
                y_pred = y_pred.drop(y_pred.index[idx_val])

            mape = mean_absolute_percentage_error(y_val, y_pred)
            scores.append(mape)
        
        # append mean MAPE scores and standart deviations overall all cross-validation folds per time series to dataframe
        cval.loc[i, 'group'] = f'{group[0]} | {group[1]}'
        cval.loc[i, 'MAPE_mean'] = np.mean(scores)
        cval.loc[i, 'MAPE_std'] = np.std(scores)
    # calculate mean scores over all time series
    cval.loc[i+1, 'group'] = 'mean'
    cval.loc[i+1, 'MAPE_mean'] = cval['MAPE_mean'].mean()
    cval.loc[i+1, 'MAPE_std'] = cval['MAPE_std'].mean()

    return cval

###################################################################################################################
# cross-validation for LightGBM corrected for holiday effects
def crossval_lgbm(df_train, grouping_vars, target, features, lgbm_kwargs=None, splits=52, test_size=7, gap=0):
    """Cross-Validation for LightGBM model:
    Takes in a training dataset of stacked time series and performs TimeSeriesSplit Cross-Validation for LightGBM model 
    for each of those time series. Returns a dataframe of cross-validation results containing mean MAPE scores 
    and corresponding standard deviations from cross-validation of each individual time series.
    
    ATTENTION: This function corrects the validation set for possible NaNs due to branch closures on holidays by dropping 
    such elements and making a prediction for a reduced validation set. However, this correction assumes such NaNs to be represented as 1. 
    Thus, any NaNs due to branch closures on holidays should be converted into 1 before applying this function.

    Args:
        df_train (pd.DataFrame): A training dataframe containing stacked time series data
        grouping_vars (list): A list of grouping variables, according to which training data is stacked. Currently accepts only a list of two variables.
        target (str): Prediction target
        features (list): List of feature names to be used for training the model
        lgbm_kwargs (dict, optional): Dictionary of LGBM hyperparameters. Defaults to None. If None, model is trained using default hyperparameters.
        splits (int, optional): Number of splits for Cross-Validation. Defaults to 52 (1 fold / week).
        test_size (int, optional): Size of validation set (i.e. forecasting horizon). Defaults to 7 days.
        gap (int, optional): Time gap between end of training and start of validation set. Defaults to 0.

    Returns:
        pd.DataFrame: A dataframe with cross-validation results, containing mean MAPE scores and respective standard deviations for each time series
    """
    # initialize dataframe for evaluation scores
    cval = pd.DataFrame({'group': [], 'MAPE_mean': [], 'MAPE_std': []})

    # iterate over all individual series and perform cross-validation
    from itertools import product
    for i, group in enumerate(product(df_train[grouping_vars[0]].unique(), df_train[grouping_vars[1]].unique())):

        # subselect time series
        ts = df_train[(df_train[grouping_vars[0]]==group[0]) & (df_train[grouping_vars[1]]==group[1])].copy()

        # perform cross validation
        tss = TimeSeriesSplit(n_splits=splits, test_size=test_size, gap=gap)
        # initialize scores list for append MAPE scores from individual folds and start cross-validation
        scores = []
        for train_i, val_i in tss.split(ts):

            train = ts.iloc[train_i]
            val = ts.iloc[val_i]

            # generate target and feature vectors
            X_train = train[features]
            X_val = val[features]
            y_train = train[target]
            y_val = val[target]

            # initialize model
            if lgbm_kwargs==None:
                lgbm = LGBMRegressor(objective='regression', random_state=42)
            else:
                lgbm = LGBMRegressor(objective='regression', random_state=42, **lgbm_kwargs)
            # train model
            lgbm.fit(X_train, y_train)
            # predict
            y_pred= pd.Series(lgbm.predict(X_val))

            # correct for holiday effects in validation set if necessary
            # if 1 (representing missing values on a holiday) is in validation set, drop elements at corresponding index position in both y_val and y_pred
            if 1 in y_val.unique():
                idx_val = [i for i in range(len(y_val.tolist())) if y_val.tolist()[i]==1]
                y_val = y_val.drop(y_val.index[idx_val])
                y_pred = y_pred.drop(y_pred.index[idx_val])

            mape = mean_absolute_percentage_error(y_val, y_pred)
            scores.append(mape)
        
        # append mean MAPE scores and standart deviations overall all cross-validation folds per time series to dataframe
        cval.loc[i, 'group'] = f'{group[0]} | {group[1]}'
        cval.loc[i, 'MAPE_mean'] = np.mean(scores)
        cval.loc[i, 'MAPE_std'] = np.std(scores)
    # calculate mean scores over all time series
    cval.loc[i+1, 'group'] = 'mean'
    cval.loc[i+1, 'MAPE_mean'] = cval['MAPE_mean'].mean()
    cval.loc[i+1, 'MAPE_std'] = cval['MAPE_std'].mean()
    
    return cval

###################################################################################################################
# feature importance for LightGBM features
def get_lgbm_feature_importance(df_train, grouping_vars, target, features, lgbm_kwargs=None, filepath=None):
    """Extracting feature importance from LightGBM model:
    Trains separate LightGBM models for individual time series in a stacked time series training dataframe and extracts feature importances.
    Returns dataframe containing feature importances for each individual time series. 
    Allows to save LGBM models for individual time series if required.

    Args:
        df_train (pd.DataFrame): A training dataframe containing stacked time series data
        grouping_vars (list): A list of grouping variables, according to which training data is stacked. Currently accepts only a list of two variables.
        target (str): Prediction target
        features (list): List of feature names to be used for training the model
        lgbm_kwargs (dict, optional): Dictionary of LGBM hyperparameters. Defaults to None. If None, model is trained using default hyperparameters.
        filepath (str, optional): File path for saving trained model. Defaults to None.

    Returns:
        pd.DataFrame: a dataframe containing feature importances for all indivudal time series.
    """
    # initialize empty dataframe with group column and feature columns
    fimportance = pd.DataFrame({}, columns=['group']+features)

    # iterate over all individual series and fit LightGBM model
    for i, group in enumerate(product(df_train[grouping_vars[0]].unique(), df_train[grouping_vars[1]].unique())):

        # subselect time series
        ts_train = df_train[(df_train[grouping_vars[0]]==group[0]) & (df_train[grouping_vars[1]]==group[1])].copy()


        X_train = ts_train[features]
        y_train = ts_train[target]

        if lgbm_kwargs!=None:
            lgbm = LGBMRegressor(objective='regression', random_state=42, importance_type='gain', **lgbm_kwargs)
        else:
            lgbm = LGBMRegressor(objective='regression', random_state=42, importance_type='gain')
        
        # train model
        lgbm.fit(X_train, y_train)
        
        # save model if filepath specified
        if filepath != None:
            lgbm.booster_.save_model(filename=os.path.join(filepath, f'lgbm_{group[0]}_{group[1]}.txt'))

        # append feature importances per time series to dataframe
        fimportance.loc[i, 'group'] = f'{group[0]} | {group[1]}'
        fimportance.loc[i, fimportance.columns[1:]] = lgbm.feature_importances_.tolist()
    # calculate mean feature importance averaged over all individual time series
    fimportance.loc[i+1, 'group'] = 'mean'
    fimportance.loc[i+1, fimportance.columns[1:]] = [fimportance[x].mean() for x in fimportance.columns[1:]]

    return fimportance

###################################################################################################################
# predict target values using LightGBM on test set for specified time window
def LGBM_predict(df_train, df_test, grouping_vars, target, features, lgbm_kwargs, start_date, end_date, compute_shap=False, plot=False, show_baseline=False):
    """Predict target values using LightGBM for defined time window:
    Fits a LightGBM model to each individual time series and generates a prediction based on a specified time window in the test dataset.
    Can accept any time window length specified by start_date and end_date. However, it is highly recommended to set the prediction time window 
    to a maximum of 7 days. Also computes shap values and plots prediction results for specified time window if specified.

    Args:
        df_train (pd.DataFrame): A training dataframe containing stacked time series data
        df_test (pd.DataFrame): A test dataframe containing stacked time series data
        grouping_vars (list): A list of grouping variables, according to which training data is stacked. Currently accepts only a list of two variables.
        target (str): Prediction target
        features (list): List of feature names to be used for training the model
        lgbm_kwargs (dict, optional): Dictionary of LGBM hyperparameters. Defaults to None. If None, model is trained using default hyperparameters.
        start_date (str): Start date of prediction time window
        end_date (str): End date of prediction time window
        compute_shap (bool, optional): Compute shap values for each prediction. Defaults to False.
        plot (bool, optional): Display time series plots for observed and predicted values. Defaults to False.
        show_baseline (bool, optional): Include predictions by Seasonal Naive baseline in time series plots. Defaults to False.

    Returns:
        dict: Dictionary containg predicted y-values (y_pred), observed y-values (y_true), MAPE scores and shap values associated with
        the predictions for each individual time series.
    """
    # initialize dataframe for evaluation scores
    preds = {'group': [], 'y_pred': [], 'y_true': [], 'MAPE': [], 'shap': []}

    # iterate over all individual series
    for i, group in enumerate(product(df_test[grouping_vars[0]].unique(), df_test[grouping_vars[1]].unique())):

        # subselect time series for train and test
        ts_test = df_test[(df_test[grouping_vars[0]]==group[0]) & (df_test[grouping_vars[1]]==group[1])].copy()
        ts_train = df_train[(df_train[grouping_vars[0]]==group[0]) & (df_train[grouping_vars[1]]==group[1])].copy()
        
        # generate target and feature vectors
        X_train = ts_train[features]
        X_test = ts_test[features]
        y_train = ts_train[target]
        y_test = ts_test[target]

        # initialize model
        if lgbm_kwargs==None:
            lgbm = LGBMRegressor(objective='regression', random_state=42)
        else:
            lgbm = LGBMRegressor(objective='regression', random_state=42, **lgbm_kwargs)
        # fit model to train data
        lgbm.fit(X_train, y_train)

        # extract 7d prediction sample and predict
        y_test_sample = y_test[(y_test.index >= pd.to_datetime(start_date)) & (y_test.index <= pd.to_datetime(end_date))]
        X_test_sample = X_test[(X_test.index >= pd.to_datetime(start_date)) & (X_test.index <= pd.to_datetime(end_date))]
        y_pred = pd.Series(lgbm.predict(X_test_sample))

        # also extract prediction from naive baseline to show for comparison if required
        y_pred_naive = y_test[(y_test.index < pd.to_datetime(start_date))]
        y_pred_naive = y_pred_naive[-7:]
        # correct for holiday effects; if holiday is in predicted y-values, replace by sales 14 days ago
        if 1 in y_pred_naive.unique():
            idx_naive = [i for i in range(len(y_pred_naive.tolist())) if y_pred_naive.tolist()[i]==1]
            idx_naive = [i-7 for i in idx_naive]
            idx_naive_lag = [i-7 for i in idx_naive]
            y_pred_naive = y_test.loc[(y_test.index < pd.to_datetime(start_date))]
            y_pred_naive.iloc[idx_naive] = [x for x in y_test.iloc[idx_naive_lag]]
            y_pred_naive = y_pred_naive[-7:]
        
        # correct for holiday effects in validation set if necessary
        # if holiday is in validation set, drop elements at corresponding index position in both y_test and y_pred
        if 1 in y_test_sample.unique():
            idx_test = [i for i in range(len(y_test_sample.tolist())) if y_test_sample.tolist()[i]==1]
            y_test_sample = y_test_sample.drop(y_test_sample.index[idx_test])
            y_pred = y_pred.drop(y_pred.index[idx_test])
            y_pred_naive = y_pred_naive.drop(y_pred_naive.index[idx_test])

        # compute MAPE
        mape = mean_absolute_percentage_error(y_test_sample, y_pred)

        # append results
        preds['group'].append(f'{group[0]} | {group[1]}')
        preds['y_pred'].append([y_pred])
        preds['y_true'].append([y_test_sample])
        preds['MAPE'].append(mape)

        # compute shap values
        if compute_shap==True:
            explainer = shap.Explainer(lgbm)
            shap_values = explainer(X_test_sample)
            preds['shap'].append(shap_values)
        
        # plot prediction results over observed values
        if plot==True:
            sample_data = y_test_sample.reset_index()
            sample_data['y_pred_LGBM'] = y_pred.values
            sample_data['y_pred_naive'] = y_pred_naive.values
            sample_data.columns = ['date', 'observed', 'y_pred_LGBM', 'y_pred_naive']
            
            plt.figure(figsize=(6, 2))
            sns.lineplot(data=sample_data, x='date', y='observed', color='black')
            # plot also baseline predictions for reference if required
            if show_baseline==True:
                sns.lineplot(data=sample_data, x='date', y='y_pred_naive', color='red', marker='o')
            sns.lineplot(data=sample_data, x='date', y='y_pred_LGBM', color='blue', marker='o')
            plt.ylabel('Turnover [€]', fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylim(0, np.max(sample_data['observed'])+100)
            plt.xlabel(None)
            plt.xticks(rotation=45, ha='right', fontsize=11)
            if show_baseline==True:
                plt.legend(labels=['observed', 'predicted by baseline', 'predicted by LGBM'], bbox_to_anchor=(1.05, 1.0), loc='upper left')
            else:
                plt.legend(labels=['observed', 'predicted'], bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.title(f'{group[0]} | {group[1]}\n{start_date} - {end_date}', fontsize=14)
            plt.show()

    return preds

###################################################################################################################
# predictions, residuals and feature contributions from LightGBM cross-validation
def get_preds_and_shaps(df_train, grouping_vars, target, features, lgbm_kwargs=None, splits=52, test_size=7, gap=0, compute_baseline=False):
    """Compute predictions, residuals, and feature contributions as assessed through SHAP values generated by LightGBM across TimeSeriesSplit Cross-Validation folds.
    Takes in a training dataset of stacked time series and performs TimeSeriesSplit Cross-Validation for LightGBM model 
    for each of those time series. Computes predicted target values and corresponding residuals
    from subsequent cross-validation folds as well as SHAP values for features used in LGBM models per individual time series. 
    Further allows to compute predictions and residuals for naive baseline model if required.

    Args:
        df_train (pd.DataFrame): A training dataframe containing stacked time series data
        grouping_vars (list): A list of grouping variables, according to which training data is stacked. Currently accepts only a list of two variables.
        target (str): Prediction target
        features (list): List of feature names to be used for training the model
        lgbm_kwargs (dict, optional): Dictionary of LGBM hyperparameters. Defaults to None. If None, model is trained using default hyperparameters.
        splits (int, optional): Number of splits for Cross-Validation. Defaults to 52 (1 fold / week).
        test_size (int, optional): Size of validation set (i.e. forecasting horizon). Defaults to 7 days.
        gap (int, optional): Time gap between end of training and start of validation set. Defaults to 0.
        compute_baseline (bool, optional): Compute predictions and residuals from naive baseline model if required. Defaults to False.

    Returns:
        dict: Dictionary containing datasets of observed and LightGBM-predicted target values and corresponding residuals 
        from subsequent cross-validation folds per time series.
    """
    # initialize dataframe for evaluation scores
    scores = {'group': [], 'predictions': [], 'shap_lgbm': [], 'features_lgbm': []}

    # iterate over all individual series and perform cross-validation
    for i, group in enumerate(product(df_train[grouping_vars[0]].unique(), df_train[grouping_vars[1]].unique())):

        # subselect time series
        ts = df_train[(df_train[grouping_vars[0]]==group[0]) & (df_train[grouping_vars[1]]==group[1])].copy()

        # perform cross validation
        tss = TimeSeriesSplit(n_splits=splits, test_size=test_size, gap=gap)
        # initialize empty dataframes for concatenating results from individual cross-validation folds
        predictions_local = pd.DataFrame({})
        shap_local = pd.DataFrame({}, columns=features)
        features_local = pd.DataFrame({}, columns=features)
        for train_i, val_i in tss.split(ts):

            train = ts.iloc[train_i]
            val = ts.iloc[val_i]

            # generate target and feature vectors
            X_train = train[features]
            X_val = val[features]
            y_train = train[target]
            y_val = val[target]

            # initialize LGBM model
            if lgbm_kwargs==None:
                lgbm = LGBMRegressor(objective='regression', random_state=42)
            else:
                lgbm = LGBMRegressor(objective='regression', random_state=42, **lgbm_kwargs)
            # train model
            lgbm.fit(X_train, y_train)
            
            # get shap values
            explainer = shap.Explainer(lgbm)
            shap_values = pd.DataFrame(explainer(X_val).values, columns=features)
            shap_local = pd.concat([shap_local, shap_values], axis=0)

            # get X feature values
            features_local = pd.concat([features_local, X_val], axis=0)

            y_pred_lgbm= pd.Series(lgbm.predict(X_val))

            if compute_baseline==True:
                # make prediction using Naive Seasonal baseline
                # correct for holiday effects in predicted values based on training set if necessary
                # if 1 (representing missing values on a holiday) is in predicted y-values, replace by sales 14 days ago
                if 1 in y_train[-7:].unique():
                    idx_train = [i for i in range(len(y_train[-7:].tolist())) if y_train[-7:].tolist()[i]==1]
                    idx_train = [i-7 for i in idx_train]
                    idx_train_lag = [i-7 for i in idx_train]
                    y_train_repl = y_train.copy()
                    y_train_repl.iloc[idx_train] = [x for x in y_train.iloc[idx_train_lag]]
                    y_pred_naive = y_train_repl[-7:]
                else:
                    y_pred_naive = y_train[-7:]

            # correct for holiday effects in validation set if necessary
            # if holiday is in validation set, drop elements at corresponding index position in both y_val and y_pred
            if 1 in y_val.unique():
                idx_val = [i for i in range(len(y_val.tolist())) if y_val.tolist()[i]==1]
                y_val = y_val.drop(y_val.index[idx_val])
                y_pred_lgbm = y_pred_lgbm.drop(y_pred_lgbm.index[idx_val])
                if compute_baseline==True:
                    y_pred_naive = y_pred_naive.drop(y_pred_naive.index[idx_val])

            # append results to local dataframe
            predictions = pd.DataFrame(y_val.copy())
            predictions.columns = ['y_true']
            predictions['y_pred_lgbm'] = y_pred_lgbm.values
            # here, we will subtract observed from predicted values, such that positive residuals correspond to over-estimation and negative residuals to underestimation
            predictions['residual_lgbm'] = predictions['y_pred_lgbm'] - predictions['y_true']
            
            if compute_baseline==True:
                predictions['y_pred_baseline'] = y_pred_naive.values
                predictions['residual_baseline'] = predictions['y_pred_baseline'] - predictions['y_true']

            predictions_local = pd.concat([predictions_local, predictions], axis=0)

        # append scores
        scores['group'].append(f'{group[0]} | {group[1]}')
        scores['predictions'].append(predictions_local)
        scores['shap_lgbm'].append(shap_local)
        scores['features_lgbm'].append(features_local)
    
    return scores