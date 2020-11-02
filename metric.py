import numpy as np
import keras.backend as K

def max_absolute_error(y_true, y_pred):
    return K.mean(K.max(K.abs(y_pred - y_true), axis=0))