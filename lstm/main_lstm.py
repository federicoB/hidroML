import numpy as np
from keras.layers import Dense, LSTM, BatchNormalization
from keras.models import Sequential

from load_input import load_input
from utils import sequentialize

import warnings
warnings.filterwarnings('ignore')

from metric import max_absolute_error

epoch = 5
batch_size = 256
dropout_ratio = 0.2

def lstm_training(train_x, train_y, val_x, val_y, sample_lenght, memory):
    memory = int(memory)
    sample_lenght = int(sample_lenght)

    regressor = Sequential([
        Dense(units=3, input_shape=(sample_lenght, train_x.shape[2])),
        LSTM(units=memory, return_sequences=True, dropout=dropout_ratio),
        BatchNormalization(),
        LSTM(units=memory, dropout=dropout_ratio),
        BatchNormalization(),
        Dense(1),
        # Activation(relu) #make output positive or zero
    ])

    # opt = SGD(lr=0.01, momentum=0.9, learning_rate=1e-18, clipvalue=0.5)
    regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=[max_absolute_error])
    regressor.build(input_shape=(train_x.shape))
    #print(regressor.summary())
    # plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # regressor = load_model("model1.h5", custom_objects={'max_absolute_error':max_absolute_error})

    history = regressor.fit(train_x, train_y, validation_data=(val_x, val_y), verbose=2, epochs=epoch, batch_size=batch_size)
    #regressor.save("model" + str(epoch) + ".h5")

    # fig1 = plt.figure(1)
    # plt.plot(history.losses, color='green', label='loss')

    step_ahead = 1
    i = step_ahead
    while True:
        predicted_level = np.array(regressor.predict(val_x))
        i -= 1
        if i == 0:
            break
        # if there is still prediction to do
        # sequentialize predicted discharge
        new_discharge, _ = sequentialize(predicted_level, sample_lenght)
        # TODO find a solution for missing rain and timeoftheyear data: https://stats.stackexchange.com/questions/265426/how-to-make-lstm-predict-multiple-time-steps-ahead
        # create new network input with sequentialize discharge but keep old rain data
        val_x = np.stack(
            (new_discharge[:, :, 0], val_x[:-new_discharge.shape[0], :, 1], val_x[:-new_discharge.shape[0], :, 2]), axis=2)

    # predicted_discharge = sc_discharge.inverse_transform(predicted_discharge)

    return history, regressor


