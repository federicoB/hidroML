import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

from load_input import load_input
from utils import sequentialize
from data_preprocessing.plotting.level_prediction import plot_level_prediction
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings('ignore')

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

#memory = 64
#sample_lenght = 48 * 3
epoch = 1
batch_size = 256
dropout_ratio = 0.2
training_data_ratio = 0.9

pbounds = {'memory': (4,256), 'sample_lenght':(4,256)}

def lstm_training(memory,sample_lenght):
    memory = int(memory)
    sample_lenght = int(sample_lenght)
    train_x, train_y, val_x, val_y, val_dates, level_start = load_input(sample_lenght, training_data_ratio)

    regressor = Sequential([
        Dense(units=3, input_shape=(sample_lenght, train_x.shape[2])),
        LSTM(units=memory, return_sequences=True, dropout=dropout_ratio),
        LSTM(units=memory, dropout=dropout_ratio),
        Dense(1),
        # Activation(relu) #make output positive or zero
    ])

    # opt = SGD(lr=0.01, momentum=0.9, learning_rate=1e-18, clipvalue=0.5)
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.build(input_shape=(train_x.shape))
    #print(regressor.summary())
    # plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # regressor = load_model("model1.h5", custom_objects={'max_absolute_error':max_absolute_error})

    regressor.fit(train_x, train_y, validation_data=(val_x, val_y), verbose=2, epochs=epoch, batch_size=batch_size)
    regressor.save("model" + str(epoch) + ".h5")

    # fig1 = plt.figure(1)
    # plt.plot(history.losses, color='green', label='loss')

    step_ahead = 1
    i = step_ahead
    while True:
        predicted_level = np.array(regressor.predict(val_x))
        i -= 1
        if i == 0:
            break
        # if there is still prediction to do
        # sequentialize predicted discharge
        new_discharge, _ = sequentialize(predicted_level, sample_lenght)
        # TODO find a solution for missing rain and timeoftheyear data: https://stats.stackexchange.com/questions/265426/how-to-make-lstm-predict-multiple-time-steps-ahead
        # create new network input with sequentialize discharge but keep old rain data
        val_x = np.stack(
            (new_discharge[:, :, 0], val_x[:-new_discharge.shape[0], :, 1], val_x[:-new_discharge.shape[0], :, 2]), axis=2)

    # predicted_discharge = sc_discharge.inverse_transform(predicted_discharge)

    x = predicted_level.flatten()
    y = val_y[memory * (step_ahead - 1):]
    difference = np.abs(x - y)
    metric = np.max(difference)
    #print("max error {:.2f} m".format(metric))
    #print("max error was on {}".format(val_dates[np.argmax(difference)]))
    return -metric

optimizer = BayesianOptimization(
    f=lstm_training,
    pbounds=pbounds,
    random_state=1
)

text = "LSTM memory {} " \
       "sample lenght {} \n" \
        "epoch {} \n" \
        "batch_size {} \n" \
        "dropout_ratio {} \n" \
        "step_ahead {} \n" \
#         .format(memory,sample_lenght,epoch,batch_size,dropout_ratio,step_ahead)

#plot_level_prediction(val_dates, predicted_level,val_y,step_ahead, text)

bayesian_opt_file = 'bayesian_optimization_lstm_log.json'
backup_file = 'bayesian_optimization_lstm_log_bck.json'
from shutil import copyfile
import os
if os.path.exists(bayesian_opt_file):
    copyfile(bayesian_opt_file,backup_file)
    load_logs(optimizer, logs=[bayesian_opt_file])

print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

logger = JSONLogger(path=backup_file)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=16,
    n_iter=18,
)

print("optimizer max")
print(optimizer.max)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))


if os.path.exists(bayesian_opt_file):
    with open(bayesian_opt_file, 'a') as outfile:
            with open(backup_file) as infile:
                outfile.write(infile.read())
    os.remove(backup_file)

