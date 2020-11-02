import os
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs

from load_input import load_input
from lstm.main_lstm import lstm_training

training_data_ratio = 0.9
sample_length = 32
step_ahead = 1
memory = 32

pbounds = {'learning_rate': (0.01,0.09), 'momentum':(0,1)}

def bayesian_wrapper(learning_rate,momentum):
    train_x, train_y, val_x, val_y, val_dates, level_start = load_input(sample_length, training_data_ratio,step_ahead)
    history, model = lstm_training(train_x,train_y,val_x,val_y,
                        sample_length,memory,step_ahead)


    x = model.predict(val_x)
    loss = np.mean(np.max(np.abs(x-val_y),axis=0))
    return -loss

optimizer = BayesianOptimization(
    f=bayesian_wrapper,
    pbounds=pbounds,
)

bayesian_opt_file = 'bayesian_optimization_lstm_log.json'
backup_file = 'bayesian_optimization_lstm_log_bck.json'
from shutil import copyfile
if os.path.exists(bayesian_opt_file):
    copyfile(bayesian_opt_file,backup_file)
    load_logs(optimizer, logs=[bayesian_opt_file])

print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

logger = JSONLogger(path=bayesian_opt_file)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=80,
    n_iter=100,
)

print("optimizer max")
print(optimizer.max)


if os.path.exists(bayesian_opt_file):
    with open(bayesian_opt_file, 'a') as outfile:
            with open(backup_file) as infile:
                outfile.write(infile.read())
    os.remove(backup_file)
