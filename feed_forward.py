from tensorflow.keras.layers import Input, Dense, LayerNormalization, subtract, add
from tensorflow.keras.models import Model


def doubleDense(units, x):
    x = Dense(units)(x)
    x = Dense(units)(x)
    return x


def get_feed_forward_model(sample_length, inner_dimension, step_ahead):
    # TODO make it as class

    def encoding(units, x):
        x = Dense(inner_dimension)(x)
        x = LayerNormalization()(x)
        x = Dense(units)(x)
        x = LayerNormalization()(x)
        return x

    in_seq = Input(shape=(sample_length, 3))
    timeof = doubleDense(sample_length, in_seq[:, :, 2])
    timeof2 = doubleDense(sample_length, in_seq[:, :, 2])
    river = encoding(step_ahead, subtract([in_seq[:, :, 0], timeof]))
    rain = encoding(step_ahead, subtract([in_seq[:, :, 1], timeof2]))
    out = add([rain, river])
    return Model(inputs=in_seq, outputs=out)
