import pandas as pd
import numpy as np
from utils import data_scale, sequentialize, split_dataset, deltalize

basin = "enza"
river_station = "compiano"
rain_station = "vetto"
start_year = 2013
end_year = 2016


def load_input(sample_lenght, training_data_ratio):
    dataset_level = pd.read_csv("data/level/{}/{}/{}-{}.csv".format(basin, river_station, start_year, end_year),
                                parse_dates=[0], index_col=0)
    dataset_rain = pd.read_csv("data/rain/{}/{}/{}-{}.csv".format(basin, rain_station, start_year, end_year),
                               parse_dates=[0], index_col=0)

    #dataset_level['level'] = dataset_level['level'].rolling(48).mean()
    #dataset_rain['Rain'] = dataset_rain['Rain'].rolling(48).mean()

    # (to save space in file) create new column as index
    dataset_level['rain'] = dataset_rain.values
    dataset_level['dayofyear'], _ = data_scale(dataset_level.index.dayofyear.values)

    dataset = dataset_level.dropna()
    dates = dataset.index.values

    dataset = dataset.values

    # TODO normalize subtracting the mean and diving for standard deviation

    # dataset_level[:, 0], sc_discharge = data_scale(dataset_level[:,0])
    # dataset_level[:,2], sc_rain = data_scale(dataset_level[:,2])

    level_start = dataset[0, 0]
    #deltalized_dataset = deltalize(dataset[:, 0:2])
    #dataset = np.append(deltalized_dataset, dataset[1:, 2].reshape(-1, 1), axis=1)

    #dataset[:, 0], _ = data_scale(dataset[:, 0])
    #dataset[:, 1], _ = data_scale(dataset[:, 1])

    x_dataset, y_dataset = sequentialize(dataset, sample_lenght)

    train_x, val_x = split_dataset(x_dataset, training_data_ratio)
    train_y, val_y = split_dataset(y_dataset, training_data_ratio)

    val_dates = dates[-val_y.shape[0]:]

    return train_x, train_y, val_x, val_y, val_dates, level_start
