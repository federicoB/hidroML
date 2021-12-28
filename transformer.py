import tensorflow as tf
from keras import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, \
    GlobalAveragePooling1D, subtract, add, Attention


def doubleDense(units, x):
    x = Dense(units)(x)
    x = Dense(units)(x)
    return x


def get_transformer_model(sample_length, step_ahead):
    # TODO make it as class

    def transformer_regression(units, x):
        query = Dense(8)(x)
        value = Dense(8)(x)
        key = Dense(8)(x)
        query, value, key = [tf.expand_dims(x, axis=1) for x in [query, value, key]]
        x = Attention()([query, value, key])
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D(data_format='channels_last')(x)
        x = Dense(units)(x)
        return x

    in_seq = Input(shape=(sample_length, 3))
    timeof = doubleDense(sample_length, in_seq[:, :, 2])
    timeof2 = doubleDense(sample_length, in_seq[:, :, 2])
    river = transformer_regression(step_ahead, subtract([in_seq[:, :, 0], timeof]))
    rain = transformer_regression(step_ahead, subtract([in_seq[:, :, 1], timeof2]))
    out = add([rain, river])
    return Model(inputs=in_seq, outputs=out)
