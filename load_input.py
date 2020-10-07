import pandas as pd
from utils import data_scale, sequentialize, split_dataset

basin = "enza"
river_station = "compiano"
rain_station = "vetto"
start_year = 2013
end_year = 2016

def load_input(sample_lenght, training_data_ratio):
    dataset_discharge = pd.read_csv("data/level/{}/{}/{}-{}.csv".format(basin,river_station,start_year,end_year),
                                    parse_dates=[0], index_col=0)
    dataset_rain = pd.read_csv("data/rain/{}/{}/{}-{}.csv".format(basin,rain_station,start_year,end_year),
                               parse_dates=[0], index_col=0)

    dataset_discharge['level'] = dataset_discharge['level'].rolling(48, center=True).mean()
    dataset_rain['Rain'] = dataset_rain['Rain'].rolling(48 * 14, center=True).mean()

    # (to save space in file) create new column as index
    dataset_discharge['dayofyear'], _ = data_scale(dataset_discharge.index.dayofyear.values)
    dataset_discharge['rain'] = dataset_rain.values

    dataset_discharge = dataset_discharge.dropna()
    dates = dataset_discharge.index.values

    dataset_discharge = dataset_discharge.values

    dataset_lenght = dataset_discharge.shape[0]

    # TODO normalize subtracting the mean and diving for standard deviation

    # dataset_discharge[:, 0], sc_discharge = data_scale(dataset_discharge[:,0])
    # dataset_discharge[:,2], sc_rain = data_scale(dataset_discharge[:,2])

    x_dataset, y_dataset = sequentialize(dataset_discharge, sample_lenght)
    dates = dates[-y_dataset.shape[0]:]

    train_x, val_x = split_dataset(x_dataset, training_data_ratio)
    train_y, val_y = split_dataset(y_dataset, training_data_ratio)

    val_dates = dates[-val_y.shape[0]:]

    return train_x, train_y, val_x, val_y, val_dates