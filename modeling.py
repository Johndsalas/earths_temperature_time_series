
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing

from math import sqrt
from sklearn.metrics import mean_squared_error

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
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat[target_var])), 4)
    print(model_type, f'-- RMSE: {rmse}')
    plt.show()

    return rmse


def last_observed_value(train, validate, target_var, eval_df):

    model_type = "Last Observed Value"

    temps = train[target_var][-1:][0]
    yhat= pd.DataFrame({target_var : [temps]}, index = validate.index)

    rmse = plot_and_eval(train, validate, test, yhat, target_var, model_type) 

    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df


def simple_average(train, validate, target_var, eval_df):

    model_type = "Simple Average"

    temps = round(train[target_var].mean(),4)
    yhat = pd.DataFrame({target_var: [temps]}, index = validate.index)

    rmse = plot_and_eval(train, validate, test, yhat, target_var, model_type)

    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df

def moving_average(train, validate, target_var, eval_df):

    index = 0

    label =['One Month', 'One Year', 'One Decade']

    for period in [1,12,120]:

        model_type = f"Moving Average {label[index]}"
     
        temps = round(train[target_var].rolling(period).mean().iloc[-1],4)
        yhat = pd.DataFrame({target_var: [temps]}, index = validate.index)
       
        rmse = plot_and_eval(train, validate, test, yhat, target_var, model_type)
       
        eval_df = append(model_type, target_var, rmse, eval_df)

        index += 1

        return eval_df

def holts(train, validate, target_var, eval_df):

    model_type = "Holt's Linear Trend"

    model = Holt(train[target_var], exponential = False)
    model = model.fit(smoothing_level = .1,
                      smoothing_slope = .1,
                      optimized = False)
    
    temps = model.predict(start = validate.index[0], end = validate.index[-1])

    yhat = pd.DataFrame({target_var: '1'}, index = validate.index)
    yhat[target_var] = round(temps,4)

    rmse = plot_and_eval(train, validate, test, yhat, target_var, model_type)
        
    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df

def next_cycle(train, train_cycle, validate, target_var, eval_df):

    yhat = train_cycle_land + train_land.diff(365).mean()

    yhat.index = validate_land.index

    rmse = plot_and_eval(train, validate, test, yhat, target_var, model_type)
        
    eval_df = append(model_type, target_var, rmse, eval_df)

    return eval_df