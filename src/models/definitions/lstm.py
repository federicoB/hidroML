import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, LSTM, subtract, add
from tensorflow.keras.models import Model


def doubleDense(units, x):
    x = Dense(units)(x)
    x = Dense(units)(x)
    return x


def get_lstm_model(sample_length, step_ahead=3, memory_units=8):
    # TODO make it as class

    def LSTM_layer_wrapper(units, x):
        x = tf.expand_dims(x, axis=-1)
        x = LSTM(units=memory_units)(x)
        x = LayerNormalization()(x)
        x = Dense(units)(x)
        return x

    in_seq = Input(shape=(sample_length, 3))
    timeof = doubleDense(sample_length, in_seq[:, :, 2])
    timeof2 = doubleDense(sample_length, in_seq[:, :, 2])
    river = LSTM_layer_wrapper(step_ahead, subtract([in_seq[:, :, 0], timeof]))
    rain = LSTM_layer_wrapper(step_ahead, subtract([in_seq[:, :, 1], timeof2]))
    out = add([river, rain])
    return Model(inputs=[in_seq], outputs=out)
