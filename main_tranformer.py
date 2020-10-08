import numpy as np
import pandas as pd
import os, datetime
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

from load_input import load_input
from utils import sequentialize
from trasformer import Time2Vector, TransformerEncoder

memory = 64
sample_lenght = 128
epoch = 1
batch_size = 32
dropout_ratio = 0.2
training_data_ratio = 0.9
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

train_x, train_y, val_x, val_y, val_dates, level_start = load_input(sample_lenght, training_data_ratio)

time_embedding = Time2Vector(sample_lenght)
attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

'''Construct model'''
in_seq = Input(shape=(sample_lenght, 3))
x = time_embedding(in_seq)
x = Concatenate(axis=-1)([in_seq, x])
x = attn_layer1((x, x, x))
x = attn_layer2((x, x, x))
x = attn_layer3((x, x, x))
x = GlobalAveragePooling1D(data_format='channels_first')(x)
x = Dropout(0.1)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(1, activation='linear')(x)

regressor = Model(inputs=in_seq, outputs=out)
regressor.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])


regressor.build(input_shape=(train_x.shape))
print(regressor.summary())
#plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#regressor = load_model("model1.h5", custom_objects={'max_absolute_error':max_absolute_error})

regressor.fit(train_x, train_y, validation_data=(val_x,val_y), verbose=2, epochs=epoch, batch_size=batch_size, shuffle=True)
regressor.save("model"+str(epoch)+".h5")

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
    # TODO find a solution for missing rain and timeoftheyear data: https://stats.stackexchange.com/questions/265426/how-to-make-lstm-predict-multiple-time-steps-ahead
    # create new network input with sequentialize discharge but keep old rain data
    val_x = np.stack((new_discharge[:,:,0], val_x[:-new_discharge.shape[0],:,1], val_x[:-new_discharge.shape[0],:,2]),axis=2)

#predicted_discharge = sc_discharge.inverse_transform(predicted_discharge)

x = predicted_discharge.flatten()
y = val_y[memory*(step_ahead-1):]
difference = np.abs(x-y)
metric = np.max(difference)
print("max error {:.2f} m".format(metric))
print("max error was on {}".format(val_dates[np.argmax(difference)]))

# Visualising the results
plt.plot(val_dates[(step_ahead-1):],val_y, color = 'green', label = 'Real discharge')
#plt.plot(val_x[:,-1,-1], color= 'blue', label='Rain')
plt.plot(val_dates[(step_ahead-1):],predicted_discharge, color = 'red', label = 'Predicted discharge')
#plt.plot(val_dates[memory*(step_ahead-1):],np.abs(x-y), color = 'red', label = 'Error')
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