import pandas as pd
import matplotlib.pyplot as plt

input_file = "data/compiano-discharge-2012-2014.csv"

df = pd.read_csv(input_file, parse_dates=[0], index_col=0)

df.plot(figsize=(2^16,2^5))

plt.show()