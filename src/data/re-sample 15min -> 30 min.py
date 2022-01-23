import os

import pandas as pd

# os.chdir("..")
os.getcwd()

# ONLY CHANGE THIS VARIABLES
data_type = "level"
basin = "reno"
station = "casalecchio-chiusa"
year = 2019

folder = "data/{}/{}/{}".format(data_type, basin, station)
df = pd.read_csv(folder + "/{}.csv".format(year),
                 skiprows=3, skipfooter=10,
                 parse_dates=[0], index_col=0)
df = df[::2]
output_name = "{}/{}-30min.csv".format(folder, year)
df.to_csv(output_name)
