from keras.layers import Dense, LSTM, BatchNormalization
from keras.models import Sequential

import warnings
warnings.filterwarnings('ignore')

from metric import sum_max_absolute_error

dropout_ratio = 0.2

def lstm_training(train_x, train_y, val_x, val_y, epoch, batch_size, sample_lenght, memory, step_ahead):
    memory = int(memory)
    sample_lenght = int(sample_lenght)

    regressor = Sequential([
        LSTM(units=memory, return_sequences=True, dropout=dropout_ratio, input_shape=(sample_lenght, train_x.shape[2])),
        BatchNormalization(),
        LSTM(units=memory, dropout=dropout_ratio),
        BatchNormalization(),
        Dense(step_ahead, activation='relu'),
    ])


    regressor.compile(optimizer='adam', loss=sum_max_absolute_error)
    regressor.build(input_shape=(train_x.shape))

    # plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    history = regressor.fit(train_x, train_y, validation_data=(val_x, val_y), verbose=2, epochs=epoch, batch_size=batch_size)


    return history, regressor


