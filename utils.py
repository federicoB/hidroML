import numpy as np

def sequentialize(arr, seq_lenght):
    early_stopping = arr.shape[0] - seq_lenght - 1
    x_dataset = np.zeros((early_stopping, seq_lenght, arr.shape[1]))
    y_dataset = np.zeros((early_stopping))
    for i in range(early_stopping):
        x_dataset[i] = arr[i:i + seq_lenght]
        y_dataset[i] = arr[i + seq_lenght + 1, 0]
    return x_dataset, y_dataset

def split_dataset(arr,ratio):
    dataset_limit = int(arr.shape[0]*ratio)
    train = arr[:dataset_limit]
    val = arr[dataset_limit:]
    return train, val