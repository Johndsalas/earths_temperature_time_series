
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing

from math import sqrt
from sklearn.metrics import mean_squared_error

from fbprophet import Prophet

def append(model_type, target_var, rmse, eval_df):
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)

    return eval_df.append(d, ignore_index = True)


def plot_and_eval(train, validate, yhat, target_var, model_type):
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat[target_var])
    plt.title(target_var)
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat[target_var])), 2)
    print(model_type, f'-- RMSE: {rmse}')
    plt.show()

    return rmse


def last_observed_value(train, validate, target_var, eval_df):

    model_type = "Last Observed Value"

    temps = train[target_var][-1:][0]
    yhat= pd.DataFrame({target_var : [temps]}, index = validate.index)

    rmse = plot_and_eval(train, validate, yhat, target_var, model_type) 

    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df


def simple_average(train, validate, target_var, eval_df):

    model_type = "Simple Average"

    temps = round(train[target_var].mean(),2)
    yhat = pd.DataFrame({target_var: [temps]}, index = validate.index)

    rmse = plot_and_eval(train, validate, yhat, target_var, model_type)

    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df

def moving_average(train, validate, target_var, eval_df):

    index = 0

    label =['One Month', 'One Year', 'One Decade']

    for period in [1,12,120]:

        model_type = f"{label[index]} Moving Average "
     
        temps = round(train[target_var].rolling(period).mean().iloc[-1],2)
        yhat = pd.DataFrame({target_var: [temps]}, index = validate.index)
       
        rmse = plot_and_eval(train, validate, yhat, target_var, model_type)
       
        eval_df = append(model_type, target_var, rmse, eval_df)

        index += 1

    return eval_df

def holt(train, validate, target_var, eval_df):

    model_type = "Holt's Linear Trend"

    model = Holt(train[target_var], exponential = False)
    model = model.fit(smoothing_level = .1,
                      smoothing_slope = .1,
                      optimized = False)
    
    temps = model.predict(start = validate.index[0], end = validate.index[-1])

    yhat = pd.DataFrame({target_var: '1'}, index = validate.index)
    yhat[target_var] = round(temps,4)

    rmse = plot_and_eval(train, validate, yhat, target_var, model_type)
        
    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df

def next_cycle(train, train_cycle, validate, target_var, eval_df):

    model_type = "Predict Next Cycle"

    yhat = train_cycle + train.diff(365).mean()

    yhat.index = validate.index

    rmse = plot_and_eval(train, validate, yhat, target_var, model_type)
        
    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df


def holt_winter(train, validate, target_var, eval_df):

    model_type = "Holt Winter"

    model = ExponentialSmoothing(np.asarray(train[target_var]) ,seasonal_periods=12 ,trend='add', seasonal='add',)

    model = model.fit()

    yhat = pd.DataFrame({target_var: '1'}, index = validate.index)
    yhat_items = model.forecast(len(yhat))
    yhat[target_var] = yhat_items

    rmse = plot_and_eval(train, validate, yhat, target_var, model_type)
        
    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df

def get_baseline(train, validate, target_var, eval_df):

    eval_df = last_observed_value(train, validate, target_var, eval_df)
    eval_df = simple_average(train, validate, target_var, eval_df)
    eval_df = moving_average(train, validate, target_var, eval_df)

    return eval_df

def get_prophet(train, validate, target_var, eval_df):

    model_type = "Prophet"

    # get training dataframe for profit
    train_prophet = train.reset_index().rename(columns = {"dt":"ds", target_var:"y"})
    train_prophet['ds'] = pd.to_datetime(train_prophet['ds'])

    # get validation range for prophet
    validate_prophet = validate.reset_index()
    validate_prophet = validate_prophet.rename(columns = {'dt':'ds'})
    validate_prophet= validate_prophet[['ds']]

    # define and fit the model
    model = Prophet()
    model.fit(train_prophet)

    # use the model to make a forecast
    forecast = model.predict(validate_prophet)

    # get variables for plot and eval and append
    yhat = forecast[['ds','yhat']]
    yhat = yhat.rename(columns={'yhat':target_var}, index = forecast.ds)

    rmse = plot_and_eval(train, validate, yhat, target_var, model_type)

    eval_df = append(model_type, target_var, rmse, eval_df)