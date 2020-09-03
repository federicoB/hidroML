import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import os
from keras.utils.vis_utils import plot_model
from tensorflow.keras import regularizers
from utils import sequentialize, split_dataset



memory = 64
batch_size = 32
dropout_ratio = 0.2
training_data_ratio = 0.9

dataset_discharge = pd.read_csv("data/discharge/compiano/compiano-discharge-2013-2016.csv",
                                parse_dates=[0], index_col=0)
dataset_rain = pd.read_csv("data/rain/vetto/vetto-rain-2013-2016.csv",
                           parse_dates=[0], index_col=0)

# (to save space in file) create new column as index
dataset_discharge['dayofyear'] = dataset_discharge.index.dayofyear
dataset_discharge['rain'] = dataset_rain.values

dates = dataset_discharge.index.values

dataset_discharge = dataset_discharge.dropna()

dataset_discharge = dataset_discharge.values

dataset_lenght = dataset_discharge.shape[0]


#sc_discharge = MinMaxScaler(feature_range=(0, 1))
#dataset_discharge[:, 0] = sc_discharge.fit_transform(dataset_discharge[:, 0].reshape(1, -1))
#sc_rain = MinMaxScaler(feature_range=(0,1))
#dataset_discharge[:,2] = sc_rain.fit_transform(dataset_discharge[:,2].reshape(1,-1))


x_dataset, y_dataset = sequentialize(dataset_discharge,memory)

train_x, val_x = split_dataset(x_dataset, training_data_ratio)
train_y, val_y = split_dataset(y_dataset, training_data_ratio)

#train_x = x_dataset[:train_dataset_limit + memory]
#train_y = y_dataset[:train_dataset_limit + memory]
#prediction_x = x_dataset[train_dataset_limit+memory:]

#real_discharge = y_dataset[train_dataset_limit+memory:]
#dates = dates[train_dataset_limit+memory:]


if os.path.exists("model.h5"):
    print("loading model")
    regressor = load_model("model.h5")
else:
    regressor = Sequential()
    regressor.add(LSTM(units=memory, return_sequences=False, dropout=dropout_ratio, input_shape=(memory,3)))
    regressor.add(Dense(1))

    #opt = SGD(lr=0.01, momentum=0.9, learning_rate=1e-18, clipvalue=0.5)
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.build(input_shape=(x_dataset.shape))
plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


history = LossHistory()

#regressor.fit(train_x, train_y, validation_data=(val_x,val_y) verbose=2, epochs=8, batch_size=batch_size)
#regressor.save("model.h5")
#print(history.losses)

#fig1 = plt.figure(1)
#plt.plot(history.losses, color='green', label='loss')

predicted_discharge = np.array(regressor.predict(val_x))
#predicted_discharge = predicted_discharge*sc_discharge.scale_


#TODO plot dates

fig2 = plt.figure(2)
# Visualising the results
plt.plot(val_y[:100], color = 'green', label = 'Real discharge')
plt.plot(val_x[:100,-1,-1], color= 'blue', label='Rain')
plt.plot(predicted_discharge[:100], color = 'red', label = 'Predicted discharge')
plt.title('Discharge Prediction')
plt.xlabel('Time')
plt.ylabel('Discharge')
plt.legend()
plt.show()