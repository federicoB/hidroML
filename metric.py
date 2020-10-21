import numpy as np
import keras.backend as K

def max_absolute_error(y_true,x_true):
    return K.max(K.abs(K.flatten(x_true)- y_true))