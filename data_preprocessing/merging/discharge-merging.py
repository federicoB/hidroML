import pandas as pd

input_list = [
    "compiano-discharge-2012",
    "compiano-discharge-2013",
    "compiano-discharge-2014"
]

input_list = ["data/"+name+".csv" for name in input_list]

output_name = "compiano-discharge-2012-2014"

df_list = [pd.read_csv(name, names=['Time','Dischange'],
                       usecols=[0,2], na_filter=True, skiprows=4, skipfooter=10,
                       parse_dates=[0], index_col=0) for name in input_list]

[print(df.size) for df in df_list]

df = pd.concat(df_list,axis=0)

df = df[~df.index.duplicated()]

print(df.size)

df.to_csv("data/"+output_name+".csv")


