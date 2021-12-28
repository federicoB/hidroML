import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from feed_forward import get_feed_forward_model
from lstm import get_lstm_model
from temporal_cn import get_tcn_model
from transformer import get_transformer_model
from utils import sequentialize, split_dataset, data_scale


def max_absolute_error(y_true, y_pred):
    return K.max(K.abs(y_pred - y_true), axis=0)[-1]


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=0)[-1]


# general hyperparameters
training_data_ratio = 0.9
sample_length = 128
step_ahead = 32
batch_size = 32
dropout_ratio = 0.2
epoch = 5

basin = "reno"
river_station = "casalecchio-chiusa"
rain_station = "vergato"
start_year = 2006
end_year = 2019


def load_input(sample_length, training_data_ratio, look_ahead):
    dataset_level = pd.read_csv("data/level/{}/{}/{}-{}.csv".format(basin, river_station, start_year, end_year),
                                parse_dates=[0], index_col=0)
    dataset_rain = pd.read_csv("data/rain/{}/{}/{}-{}.csv".format(basin, rain_station, start_year, end_year),
                               parse_dates=[0], index_col=0)

    # (to save space in file) create new column as index
    dataset_level['rain'] = dataset_rain.values

    # rolling average on rain and level to remove noise
    dataset_level = dataset_level.rolling(24).mean().fillna(method='bfill')

    dataset_level = dataset_level.dropna()

    dataset_level['level'], _ = data_scale(dataset_level['level'].values)
    dataset_level['rain'], _ = data_scale(dataset_level['rain'].values)
    dataset_level['dayofyear'], _ = data_scale(dataset_level.index.dayofyear.values)

    dates = dataset_level.index.values

    dataset = dataset_level.values

    x_dataset, y_dataset = sequentialize(dataset, sample_length, look_ahead)

    train_x, val_x = split_dataset(x_dataset, training_data_ratio)
    train_y, val_y = split_dataset(y_dataset, training_data_ratio)

    val_dates = dates[-val_y.shape[0]:]

    return train_x, train_y, val_x, val_y, val_dates


train_x, train_y, val_x, val_y, val_dates = \
    load_input(sample_length, training_data_ratio, step_ahead)

memory = 8

lstm_model = get_lstm_model(sample_length, memory, step_ahead)
lstm_model.compile(loss='mse', optimizer='adam', metrics=[max_absolute_error, mean_absolute_error])
lstm_model.build(input_shape=train_x.shape)
plot_model(lstm_model, show_layer_names=False, show_shapes=True)

lstm_model.summary()

early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=2, restore_best_weights=True)
history = lstm_model.fit(train_x, train_y,
                         validation_data=(val_x, val_y),
                         epochs=epoch, batch_size=batch_size, callbacks=[early_stopping])

### feed forward

inner_dimension = 8

ff_model = get_feed_forward_model(sample_length, inner_dimension, step_ahead)
ff_model.compile(loss='mse', optimizer='adam', metrics=[max_absolute_error, mean_absolute_error])
ff_model.build(input_shape=train_x.shape)
plot_model(ff_model, show_layer_names=False, show_shapes=True)

early_stopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=3, restore_best_weights=True)
history = ff_model.fit(train_x, train_y,
                       validation_data=(val_x, val_y),
                       epochs=epoch, batch_size=batch_size, callbacks=[early_stopping])
## TCN


inner_encoding = 8

tcn_model = get_tcn_model(sample_length, inner_encoding, step_ahead)
tcn_model.compile(loss='mse', optimizer='adam', metrics=[max_absolute_error, mean_absolute_error])
tcn_model.build(input_shape=train_x.shape)
plot_model(tcn_model, show_layer_names=False, show_shapes=True)

early_stopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=3, restore_best_weights=True)
history = tcn_model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=epoch, batch_size=batch_size, callbacks=[early_stopping])

## Transformer


transformer_model = get_transformer_model(sample_length, step_ahead)
transformer_model.compile(loss='mse', optimizer='adam', metrics=[max_absolute_error, mean_absolute_error])
transformer_model.build(input_shape=train_x.shape)
plot_model(transformer_model, show_layer_names=False, show_shapes=True)

early_stopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=3, restore_best_weights=True)
history = transformer_model.fit(train_x, train_y,
                                validation_data=(val_x, val_y),
                                epochs=epoch, batch_size=batch_size, callbacks=[early_stopping])

predict_level_trasformer = transformer_model.predict(val_x)
predict_level_tcn = tcn_model.predict(val_x)
predict_level_ff = ff_model.predict(val_x)
predict_level_lstm = lstm_model.predict(val_x)

max_error_tr = np.max(np.abs(predict_level_trasformer - val_y), axis=0)
max_error_tcn = np.max(np.abs(predict_level_tcn - val_y), axis=0)
max_error_ff = np.max(np.abs(predict_level_ff - val_y), axis=0)
max_error_lstm = np.max(np.abs(predict_level_lstm - val_y), axis=0)
print("max error trasformer {}".format(max_error_tr[-1]))
print("max error tcn {}".format(max_error_tcn[-1]))
print("max error ff {}".format(max_error_ff[-1]))
print("max error lstm {}".format(max_error_lstm[-1]))

obs = 21765
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_ff[obs], label='predicted_ff')
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_trasformer[obs],
         label='predicted_transformer')
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_tcn[obs], label='predicted_tcn')
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_lstm[obs], label='predicted_lstm')
plt.plot(list(range(sample_length, sample_length + step_ahead)), val_y[obs], label='truth')
plt.plot(list(range(sample_length)), val_x[obs, :, 0], label='past_river')
plt.plot(list(range(sample_length + step_ahead)), np.concatenate((val_x[obs, :, 1], val_x[obs + 1, :step_ahead, 1])),
         label='rain')
plt.legend()
plt.show()
