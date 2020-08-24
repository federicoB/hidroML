import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt


memory = 27
dropout_ratio = 0.2
training_data_ratio = 0.9

dataset = pd.read_csv("data/discharge/compiano/compiano-discharge-2013-2014.csv",
                      parse_dates=[0], index_col=0).values

sc = MinMaxScaler(feature_range=(0,1))
dataset_scaled = sc.fit_transform(dataset)

dataset_lenght = dataset_scaled.shape[0]

train_dataset_limit = int(dataset_lenght * training_data_ratio)
real_discharge = dataset_scaled[train_dataset_limit:]

x_dataset = []
y_dataset = []
for i in range(memory, dataset_scaled.shape[0]):
    x_dataset.append(dataset_scaled[i - memory:i, 0])
    y_dataset.append(dataset_scaled[i, 0])
x_dataset, y_dataset = np.array(x_dataset), np.array(y_dataset)

# Reshaping
x_dataset = np.reshape(x_dataset,(x_dataset.shape[0], x_dataset.shape[1], 1))

train_x = x_dataset[:train_dataset_limit + memory]
train_y = y_dataset[:train_dataset_limit + memory]
prediction_x = x_dataset[train_dataset_limit+memory:]

regressor = Sequential({
    LSTM(units=memory, return_sequences=True, input_shape=(1, memory)),
    Dropout(dropout_ratio),
    LSTM(units=memory, return_sequences=True),
    Dropout(dropout_ratio),
    LSTM(units=memory, return_sequences=True),
    Dropout(dropout_ratio),
    LSTM(units=memory),
    Dropout(dropout_ratio),
    Dense(1)
})

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(train_x, train_y, epochs=1)

predicted_discharge = regressor.predict(prediction_x)

# Visualising the results
plt.plot(real_discharge, color = 'red', label = 'Real discharge')
plt.plot(predicted_discharge, color = 'blue', label = 'Predicted discharge')
plt.title('Discharge Prediction')
plt.xlabel('Time')
plt.ylabel('Discharge')
plt.legend()
plt.show()