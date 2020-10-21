import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disable tensorflow INFO logs
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

print('Tensorflow version: {}'.format(tf.__version__))

import warnings
warnings.filterwarnings('ignore')

from load_input import load_input
from utils import sequentialize
from transformer.trasformer import Time2Vector, TransformerEncoder
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import talos

from transformer.main_tranformer import trasformer_training


training_data_ratio = 0.9

params = {
    'sample_lenght':[144],
    'd_k': list(range(121,124,1)),
    'd_v': list(range(46,53,1)),
    'ff_dim': list(range(217,220,1)),
    'n_heads' : [1]
}

def grid_search_wrapper(train_x, train_y, x_val, y_val, params):
    return trasformer_training(train_x, train_y, x_val, y_val,
        params['sample_lenght'],
        params['d_k'],
        params['d_v'],
        params['n_heads'],
        params['ff_dim']
    )

train_x, train_y, val_x, val_y, val_dates, level_start = load_input(params['sample_lenght'][0], training_data_ratio)

talos.Scan(train_x,train_y,x_val=val_x, y_val=val_y, model= grid_search_wrapper, params=params, experiment_name='experiment')