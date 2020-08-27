import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import os
from tensorflow.keras import regularizers


memory = 4
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
# TODO symplifing going forward
for i in range(dataset_scaled.shape[0]-memory-1):
    #TODO check on precision lost
    x_dataset.append(dataset_scaled[i:i+memory, 0])
    y_dataset.append(dataset_scaled[i+memory+1, 0])
x_dataset, y_dataset = np.array(x_dataset), np.array(y_dataset)

# Reshaping
x_dataset = np.reshape(x_dataset,(x_dataset.shape[0], x_dataset.shape[1], 1))

train_x = x_dataset[:train_dataset_limit + memory]
train_y = y_dataset[:train_dataset_limit + memory]
prediction_x = x_dataset[train_dataset_limit+memory:]

if os.path.exists("model.h5"):
    regressor = load_model("model.h5")
else:
    regressor = Sequential({
        LSTM(units=memory, return_sequences=False, dropout=dropout_ratio, input_dim=1),
        Dense(1)
    })

    opt = SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    regressor.compile(optimizer=opt, loss='mean_absolute_percentage_error')
    #print(regressor.summary())

    print("Shape of training x: {}".format(train_x.shape))
    print("Shape of training y: {}".format(train_y.shape))


    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))


    history = LossHistory()

    regressor.fit(train_x, train_y, epochs=0, batch_size=10, callbacks=[history])
    print(history.losses)
    #regressor.save("model.h5")

predicted_discharge = regressor.predict(prediction_x)
predicted_discharge = sc.inverse_transform(predicted_discharge)

# Visualising the results
plt.plot(real_discharge, color = 'red', label = 'Real discharge')
plt.plot(predicted_discharge, color = 'blue', label = 'Predicted discharge')
plt.title('Discharge Prediction')
plt.xlabel('Time')
plt.ylabel('Discharge')
plt.legend()
plt.show()