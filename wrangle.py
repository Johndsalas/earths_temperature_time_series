
import pandas as pd

import matplotlib.pyplot as plt

def cel_to_fah(value):
    
    return (value*2) + 30


def pre_split_prep():

    # read in data
    df = pd.read_csv('GlobalTemperatures.csv')

    # set to datetime
    df.dt = pd.to_datetime(df.dt)

    # Sort rows by date then set the index as the date
    df = df.set_index("dt").sort_index()

    # make dataframe for land only and land and ocean temperatures
    df_land = df[["LandAverageTemperature"]]
    df_ocean = df[["LandAndOceanAverageTemperature"]]

    # convert temperature to fahrenheit
    df_land = df_land.apply(cel_to_fah)
    df_ocean = df_ocean.apply(cel_to_fah)

    # remove nulls
    df_land = df_land.loc[df.index > '1752-09-01']
    df_ocean = df_ocean.loc[df.index >= '1850-01-01']

    return df_land, df_ocean

def tvt_split(df_land,df_ocean):

    train_land = df_land[:'1962-12-01'] 
    validate_land = df_land['1963-01-01':'1997-12-01']
    test_land = df_land['1998-01-01':]

    train_ocean = df_ocean[:'1982-12-01']
    validate_ocean = df_ocean['1983-01-01':'2004-12-01']
    test_ocean = df_ocean['2005-01-01':]

    return train_land, validate_land, test_land, train_ocean, validate_ocean, test_ocean


def show_split(train_land, validate_land, test_land, train_ocean, validate_ocean, test_ocean):

    plt.figure(figsize = (12,4))
    plt.title("Average Land Temperature Train, Validate, and Test Data")
    plt.plot(train_land.index, train_land.LandAverageTemperature)
    plt.plot(validate_land.index, validate_land.LandAverageTemperature)
    plt.plot(test_land.index, test_land.LandAverageTemperature)
    plt.show()

    plt.figure(figsize = (12,4))
    plt.title("Average Land and Ocean Temperature Train, Validate, and Test Data")
    plt.plot(train_ocean.index, train_ocean.LandAndOceanAverageTemperature)
    plt.plot(validate_ocean.index, validate_ocean.LandAndOceanAverageTemperature)
    plt.plot(test_ocean.index, test_ocean.LandAndOceanAverageTemperature)
    plt.show()