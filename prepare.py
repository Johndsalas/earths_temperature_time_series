
import pandas as pd

def cel_to_fah(value):
    
    return (value*2) + 30


def wrangle():

    df = pd.read_csv('GlobalTemperatures.csv')

    # Reassign the sale_date column to be a datetime type
    df.dt = pd.to_datetime(df.dt)

    # Sort rows by the date and then set the index as that date
    df = df.set_index("dt").sort_index()

    df_land = df[["LandAverageTemperature"]]

    df_lo = df[["LandAndOceanAverageTemperature"]]

    df_lo = df_lo.loc[df.index >= '1850-01-01']

    df_land = df_land.apply(cel_to_fah)
    df_lo = df_lo.apply(cel_to_fah)

    df_land = df_land.loc[df.index > '1752-09-01']

    return df_land, df_lo


df_land, df_lo = wrangle()

print(df_land)