import tensorflow as tf
from keras import Model
from tcn import TCN
from tensorflow.keras.layers import Input, Dense, LayerNormalization, subtract, add


def doubleDense(units, x):
    x = Dense(units)(x)
    x = Dense(units)(x)
    return x


def get_tcn_model(sample_length, step_ahead=3, inner_encoding=8):
    # TODO make it as class

    def TCN_layer(units, x):
        x = tf.expand_dims(x, axis=-1)
        x = TCN(inner_encoding, kernel_size=2, dilations=[1, 2, 4], use_skip_connections=True)(x)
        x = LayerNormalization()(x)
        x = Dense(units)(x)
        return x

    in_seq = Input(shape=(sample_length, 3))
    timeof = doubleDense(sample_length, in_seq[:, :, 2])
    timeof2 = doubleDense(sample_length, in_seq[:, :, 2])
    river = TCN_layer(step_ahead, subtract([in_seq[:, :, 0], timeof]))
    rain = TCN_layer(step_ahead, subtract([in_seq[:, :, 1], timeof2]))
    out = add([rain, river])
    return Model(inputs=in_seq, outputs=out)
