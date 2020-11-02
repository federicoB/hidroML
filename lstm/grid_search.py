import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disable tensorflow INFO logs
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

print('Tensorflow version: {}'.format(tf.__version__))

import warnings
warnings.filterwarnings('ignore')

from load_input import load_input
import talos
# split GPU memory in two for two parallel jobs
talos.utils.gpu_utils.parallel_gpu_jobs(0.5)

from lstm.main_lstm import lstm_training


training_data_ratio = 0.9

params = {
    'sample_lenght':[32],
    'memory':[32],
    'look_ahead':[6]
}

def grid_search_wrapper(train_x, train_y, x_val, y_val, params):
    train_x, train_y, x_val, y_val, val_dates, level_start = load_input(params['sample_lenght'], training_data_ratio, params['look_ahead'])
    return lstm_training(train_x, train_y, x_val, y_val,
        params['sample_lenght'],
        params['memory'],
        params['look_ahead'],
    )

train_x, train_y, val_x, val_y, val_dates, level_start = load_input(params['sample_lenght'][0], training_data_ratio,  params['look_ahead'][0])

talos.Scan(train_x,train_y,x_val=val_x, y_val=val_y, model= grid_search_wrapper, params=params, experiment_name='lstm_look_ahead')