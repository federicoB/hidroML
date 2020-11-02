import numpy as np
from sklearn.preprocessing import MinMaxScaler

def deltalize(arr):
    arr = arr[1:] - arr[:-1]
    return arr

def sequentialize(arr, seq_lenght, look_ahead):
    early_stopping = arr.shape[0] - seq_lenght - look_ahead + 1
    x_dataset = np.zeros((early_stopping, seq_lenght, arr.shape[1]))
    y_dataset = np.zeros((early_stopping,look_ahead))
    for i in range(early_stopping):
        x_dataset[i] = arr[i:i + seq_lenght]
        y_dataset[i] = arr[i + seq_lenght:i+seq_lenght+look_ahead, 0]
    return x_dataset, y_dataset

def split_dataset(arr,ratio):
    dataset_limit = int(arr.shape[0]*ratio)
    train = arr[:dataset_limit]
    val = arr[dataset_limit:]
    return train, val

def data_scale(arr):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(arr.reshape(-1,1))
    new_arr = arr / (scaler.data_max_ - scaler.data_min_)
    return new_arr, scaler

def data_de_scale(arr,scaler):
    return None