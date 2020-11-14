import keras.backend as K

def max_absolute_error(y_true, y_pred):
    return K.max(K.abs(y_pred - y_true), axis=0)[-1]

def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=0)[-1]