import pandas as pd

# ONLY CHANGE THIS VARIABLES
data_type = "level"
basin = "enza"
station = "compiano"
start_year = 2013
end_year = 2016

folder = "data/{}/{}/{}".format(data_type,basin,station)
input_list = ["{}/{}.csv".format(folder,year) for year in range(start_year,end_year+1)]



df_list = [pd.read_csv(name, names=['Time',data_type],
                       usecols=[0,2], na_filter=True, skiprows=4, skipfooter=10,
                       parse_dates=[0], index_col=0) for name in input_list]

[print(df.size) for df in df_list]

df = pd.concat(df_list,axis=0)

df = df[~df.index.duplicated()]

print(df.size)

output_name = "{}/{}-{}.csv".format(folder,start_year,end_year)
df.to_csv(output_name)


