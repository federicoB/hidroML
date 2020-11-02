from keras.layers import Dense, LSTM, BatchNormalization
from keras.models import Sequential

import warnings
warnings.filterwarnings('ignore')

from metric import max_absolute_error

epoch = 200
batch_size = 256
dropout_ratio = 0.2

def lstm_training(train_x, train_y, val_x, val_y, sample_lenght, memory, step_ahead):
    memory = int(memory)
    sample_lenght = int(sample_lenght)

    regressor = Sequential([
        Dense(units=3, input_shape=(sample_lenght, train_x.shape[2])),
        LSTM(units=memory, return_sequences=True, dropout=dropout_ratio),
        BatchNormalization(),
        LSTM(units=memory, dropout=dropout_ratio),
        BatchNormalization(),
        Dense(step_ahead),
    ])


    regressor.compile(optimizer='adam', loss=max_absolute_error,metrics=[max_absolute_error])
    regressor.build(input_shape=(train_x.shape))

    # plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    history = regressor.fit(train_x, train_y, validation_data=(val_x, val_y), verbose=2, epochs=epoch, batch_size=batch_size)


    return history, regressor


