import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation
from keras.activations import relu
import matplotlib.pyplot as plt
import os
from keras.utils.vis_utils import plot_model
from tensorflow.keras import regularizers
from utils import sequentialize, split_dataset, data_scale

memory = 48
sample_lenght = memory*3
epoch = 3
batch_size = 256
dropout_ratio = 0.1
training_data_ratio = 0.9

dataset_discharge = pd.read_csv("data/discharge/compiano/compiano-discharge-2013-2016.csv",
                                parse_dates=[0], index_col=0)
dataset_rain = pd.read_csv("data/rain/vetto/vetto-rain-2013-2016.csv",
                           parse_dates=[0], index_col=0)

# (to save space in file) create new column as index
dataset_discharge['dayofyear'], _ = data_scale(dataset_discharge.index.dayofyear.values)
dataset_discharge['rain'] = dataset_rain.values

dataset_discharge = dataset_discharge.dropna()
dates = dataset_discharge.index.values

dataset_discharge = dataset_discharge.values

dataset_lenght = dataset_discharge.shape[0]

# TODO normalize subtracting the mean and diving for standard deviation

#dataset_discharge[:, 0], sc_discharge = data_scale(dataset_discharge[:,0])
#dataset_discharge[:,2], sc_rain = data_scale(dataset_discharge[:,2])


x_dataset, y_dataset = sequentialize(dataset_discharge,sample_lenght)
dates = dates[-y_dataset.shape[0]:]

train_x, val_x = split_dataset(x_dataset, training_data_ratio)
train_y, val_y = split_dataset(y_dataset, training_data_ratio)
val_dates = dates[-val_y.shape[0]:]

model_file_name = "model"+str(epoch)+".h5"
if os.path.exists(model_file_name):
    print("loading model")
    regressor = load_model(model_file_name)
else:
    regressor = Sequential([
        Dense(units=3,input_shape=(sample_lenght,dataset_discharge.shape[1])),
        LSTM(units=memory, return_sequences=True,dropout=dropout_ratio, ),
        LSTM(units=memory, dropout=dropout_ratio),
        Dense(1),
        Activation(relu) #make output positive or zero
    ])


    #opt = SGD(lr=0.01, momentum=0.9, learning_rate=1e-18, clipvalue=0.5)
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.build(input_shape=(x_dataset.shape))
    print(regressor.summary())
#plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


history = LossHistory()
regressor.fit(train_x, train_y, validation_data=(val_x,val_y), verbose=2, epochs=epoch, batch_size=batch_size, shuffle=True)
regressor.save(model_file_name)
#print(history.losses)

#fig1 = plt.figure(1)
#plt.plot(history.losses, color='green', label='loss')

step_ahead = 1
i = step_ahead
while True:
    predicted_discharge = np.array(regressor.predict(val_x))
    i -= 1
    if i == 0:
        break
    # if there is still prediction to do
    # sequentialize predicted discharge
    new_discharge, _ = sequentialize(predicted_discharge,sample_lenght)
    # TODO find a solution for missing rain and timeoftheyear data
    # create new network input with sequentialize discharge but keep old rain data
    val_x = np.stack((new_discharge[:,:,0], val_x[:-new_discharge.shape[0],:,1], val_x[:-new_discharge.shape[0],:,2]),axis=2)

#predicted_discharge = sc_discharge.inverse_transform(predicted_discharge)

# Visualising the results
plt.plot(val_dates[memory*(step_ahead-1):],val_y[memory*(step_ahead-1):], color = 'green', label = 'Real discharge')
#plt.plot(val_x[:,-1,-1], color= 'blue', label='Rain')
plt.plot(val_dates[memory*(step_ahead-1):],predicted_discharge, color = 'red', label = 'Predicted discharge')
plt.text(0,1,"LSTM memory {} \n"
        "sample lenght {} \n"
        "epoch {} \n"
        "batch_size {} \n"
        "dropout_ratio {} \n"
        "step_ahead {} \n"
         .format(memory,sample_lenght,epoch,batch_size,dropout_ratio,step_ahead),
         transform = plt.gca().transAxes)
plt.title('Discharge Prediction')
plt.xlabel('Time')
plt.ylabel('Discharge (m^3 / s)')
plt.legend()
plt.show()