import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disable tensorflow INFO logs
import tensorflow as tf

print('Tensorflow version: {}'.format(tf.__version__))

import warnings
warnings.filterwarnings('ignore')

from load_input import load_input
import talos
try:
    from talos.utils.gpu_utils import parallel_gpu_jobs
    # split GPU memory in two for two parallel jobs
    parallel_gpu_jobs(0.5)
except:
    pass


from transformer.main_transformer import transformer_training


training_data_ratio = 0.9

params = {
    'epoch':[1],
    'batch_size':[64],
    'sample_lenght':[256],
    'd_k': list(range(121,124,1)),
    'd_v': list(range(46,53,1)),
    'ff_dim': list(range(217,220,1)),
    'n_heads' : [1],
    'look_ahead': [1]
}

def grid_search_wrapper(train_x, train_y, x_val, y_val, params):
    return transformer_training(train_x, train_y, x_val, y_val,
                                params['epoch'],
                                params['batch_size'],
                                params['sample_lenght'],
                                params['d_k'],
                                params['d_v'],
                                params['n_heads'],
                                params['ff_dim'],
                                params['look_ahead']
                                )

train_x, train_y, val_x, val_y, val_dates, level_start = load_input(params['sample_lenght'][0], training_data_ratio, params['look_ahead'][0])

talos.Scan(train_x,train_y,x_val=val_x, y_val=val_y, model= grid_search_wrapper, params=params, experiment_name='trasformer_hyperameter')