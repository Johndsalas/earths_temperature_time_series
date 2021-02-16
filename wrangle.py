
import pandas as pd

import matplotlib.pyplot as plt

def cel_to_fah(value):
    
    return (value*2) + 30


def pre_split_prep(df):

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
    df_land = df_land.loc[df.index >= '1753']
    df_ocean = df_ocean.loc[df.index >= '1850']

    return df_land, df_ocean

def tvt_split(df_land,df_ocean):

    train_land = df_land[:'1963'] 
    validate_land = df_land['1964':'1997']
    test_land = df_land['1998':]

    train_ocean = df_ocean[:'1982']
    validate_ocean = df_ocean['1983':'2005']
    test_ocean = df_ocean['2006':]

    show_split(train_land, validate_land, test_land, train_ocean, validate_ocean, test_ocean)

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