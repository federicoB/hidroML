import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disable tensorflow INFO logs
import warnings
warnings.filterwarnings('ignore')
from transformer.trasformer import Time2Vector, TransformerEncoder
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Model

from tensorflow.keras.utils import plot_model

from metric import sum_max_absolute_error


dropout_ratio = 0.2

def transformer_training(train_x, train_y, val_x, val_y, epoch, batch_size,
                         sample_length, d_k, d_v, n_heads, ff_dim, step_ahead):
    TimeEmbeddingLayerRain = Time2Vector(sample_length)
    TimeEmbeddingLayerRiver = Time2Vector(sample_length)
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = Input(shape=(sample_length, 3))
    river_time_embedding = TimeEmbeddingLayerRiver(in_seq[:,:,0])
    rain_time_embedding = TimeEmbeddingLayerRain(in_seq[:,:,1]) #output features 2: period and nonperiodic time embedding
    x = Concatenate(axis=-1)([in_seq, river_time_embedding, rain_time_embedding])
    # triplicate input (query,key,value)
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    # No decoder but regression
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(step_ahead, activation='relu')(x)

    regressor = Model(inputs=in_seq, outputs=out)
    regressor.compile(loss=sum_max_absolute_error, optimizer='adam')
    regressor.build(input_shape=(train_x.shape))

    plot_model(regressor,'transformer.png',show_shapes=True, expand_nested=True)

    history = regressor.fit(train_x, train_y,
                            validation_data=(val_x,val_y),
                            epochs=epoch, batch_size=batch_size, shuffle=True)

    return history, regressor
