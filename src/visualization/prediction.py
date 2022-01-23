import numpy as np
import matplotlib.pyplot as plt
from src.models.predict_model import get_prediction
from src.data.make_dataset import load_input
from src.models.train_model import sample_length, training_data_ratio, step_ahead

# TODO loading input is already done in predict_model
train_x, train_y, val_x, val_y, val_dates = \
    load_input(sample_length, training_data_ratio, step_ahead)

predict_level_trasformer = get_prediction('transformer')
predict_level_tcn = get_prediction('tcn')
predict_level_ff = get_prediction('ff')
predict_level_lstm = get_prediction('lstm')

max_error_tr = np.max(np.abs(predict_level_trasformer - val_y), axis=0)
max_error_tcn = np.max(np.abs(predict_level_tcn - val_y), axis=0)
max_error_ff = np.max(np.abs(predict_level_ff - val_y), axis=0)
max_error_lstm = np.max(np.abs(predict_level_lstm - val_y), axis=0)
print("max error trasformer {}".format(max_error_tr[-1]))
print("max error tcn {}".format(max_error_tcn[-1]))
print("max error ff {}".format(max_error_ff[-1]))
print("max error lstm {}".format(max_error_lstm[-1]))

obs = 21765
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_ff[obs], label='predicted_ff')
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_trasformer[obs],
         label='predicted_transformer')
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_tcn[obs], label='predicted_tcn')
plt.plot(list(range(sample_length, sample_length + step_ahead)), predict_level_lstm[obs], label='predicted_lstm')
plt.plot(list(range(sample_length, sample_length + step_ahead)), val_y[obs], label='truth')
plt.plot(list(range(sample_length)), val_x[obs, :, 0], label='past_river')
plt.plot(list(range(sample_length + step_ahead)), np.concatenate((val_x[obs, :, 1], val_x[obs + 1, :step_ahead, 1])),
         label='rain')
plt.legend()
plt.show()
