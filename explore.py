
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm

def plot_land_exploration(train_land):

    plt.figure(figsize = (12,4))
    plt.title("Average Land Temperature by Month")
    plt.plot(train_land)
    plt.show

    plt.figure(figsize = (12,4))
    plt.title("Average Land Temperature by Year")
    plt.plot(train_land.resample('y').mean())
    plt.show

    plt.figure(figsize = (12,4))
    plt.title("Average Land Temperature by Decade")
    plt.plot(train_land.resample('10y').mean())
    plt.show


def plot_land_ocean_explore(train_ocean):

    plt.figure(figsize = (12,4))
    plt.title("Average Temperature by Month")
    plt.plot(train_ocean)
    plt.show

    plt.figure(figsize = (12,4))
    plt.title("Average Temperature by Year")
    plt.plot(train_ocean.resample('y').mean())
    plt.show

    plt.figure(figsize = (12,4))
    plt.title("Average Temperature by Decade")
    plt.plot(train_ocean.resample('10y').mean())
    plt.show


def compare_plot(train_land, train_ocean):

    plt.figure(figsize = (12,4))
    plt.title("Average Temperature by Month")
    plt.plot(train_land)
    plt.plot(train_ocean)
    plt.legend(["Land Only", "Land and Ocean"])
    plt.show

    plt.figure(figsize = (12,4))
    plt.title("Average Temperature by Year")
    plt.plot(train_land.resample('y').mean())
    plt.plot(train_ocean.resample('y').mean())
    plt.legend(["Land Only", "Land and Ocean"],loc='upper left')
    plt.show

    plt.figure(figsize = (12,4))
    plt.title("Average Temperature by Decade")
    plt.plot(train_land.resample('10y').mean())
    plt.plot(train_ocean.resample('10y').mean())
    plt.legend(["Land Only", "Land and Ocean"],loc='upper left')
    plt.show

def show_sesonal_decomp(train_land, validate_land, test_land, train_ocean, validate_ocean, test_ocean):

    col = 'LandAverageTemperature'
    print(col,'\n')
    _ = sm.tsa.seasonal_decompose(train_land[col].resample('M').mean()).plot()
    plt.show()

    col = 'LandAndOceanAverageTemperature'
    print(col,'\n')
    _ = sm.tsa.seasonal_decompose(train_ocean[col].resample('M').mean()).plot()
    plt.show()