import pandas as pd
import matplotlib.pyplot as plt

basin = "enza"
river_station = "compiano"
rain_station = "vetto"
start_year = 2013
end_year = 2016

input_file_level = "data/level/{}/{}/{}-{}.csv".format(basin, river_station, start_year, end_year)
input_file_rain = "data/rain/{}/{}/{}-{}.csv".format(basin,rain_station,start_year,end_year)

df_discharge = pd.read_csv(input_file_level, parse_dates=[0], index_col=0)
df_rain = pd.read_csv(input_file_rain, parse_dates=[0], index_col=0)

df = pd.concat([df_discharge,df_rain],axis=1)

df['level'] = df['level'].rolling(48,center=True).mean()
df['Rain'] = df['Rain'].rolling(48*14,center=True).mean()

fig, axs = plt.subplots(2, figsize=(2 ^ 16, 2 ^ 10))

color = 'tab:green'
axs[0].set_xlabel('date')
axs[0].set_ylabel('level m', color=color)
axs[0].plot(df.index, df['level'], color=color)
axs[0].tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
axs[1].set_ylabel('rain kg/m^2', color=color)  # we already handled the x-label with ax1
axs[1].plot(df.index, df['Rain'], color=color)
axs[1].tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()