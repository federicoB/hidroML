import keras.backend as K

def sum_max_absolute_error(y_true, y_pred):
    return K.sum(K.max(K.abs(y_pred - y_true), axis=0))