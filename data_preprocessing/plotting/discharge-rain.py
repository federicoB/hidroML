import pandas as pd
import matplotlib.pyplot as plt

input_file_discharge = "data/compiano-discharge-2014-edit.csv"
input_file_rain = "data/ramiseto-rain-2014-edit.csv"

df_discharge = pd.read_csv(input_file_discharge, parse_dates=[0], index_col=0)
df_rain = pd.read_csv(input_file_rain, parse_dates=[0], index_col=0)

df = pd.concat([df_discharge,df_rain],axis=1)

#df['Dischange'] = df['Dischange'].rolling(48,center=True).mean()
#df['Rain'] = df['Rain'].rolling(48*14,center=True).mean()

fig, axs = plt.subplots(2, figsize=(2 ^ 16, 2 ^ 10))

color = 'tab:green'
axs[0].set_xlabel('date')
axs[0].set_ylabel('discharge m^3/s', color=color)
axs[0].plot(df.index, df['Discharge'], color=color)
axs[0].tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
axs[1].set_ylabel('rain kg/m^2', color=color)  # we already handled the x-label with ax1
axs[1].plot(df.index, df['Rain'], color=color)
axs[1].tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()