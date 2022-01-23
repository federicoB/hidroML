rom
src.models.train_model
import sample_length, training_data_ratio, step_ahead, \
    max_absolute_error, mean_absolute_error
from src.data.make_dataset import load_input
# TODO load model from file
import argparse
from tensorflow import keras
# TODO load model from file
import argparse

from tensorflow import keras

from src.data.make_dataset import load_input
from src.models.train_model import sample_length, training_data_ratio, step_ahead, \
    max_absolute_error, mean_absolute_error


def get_prediction(model_name):
    # TODO make these parameters parametric in function and file
    train_x, train_y, val_x, val_y, val_dates = \
        load_input(sample_length, training_data_ratio, step_ahead)
    # TODO custom_objects hardcoded, change
    model = keras.models.load_model("models/{}".format(model_name),
                                    custom_objects={'max_absolute_error': max_absolute_error,
                                                    'mean_absolute_error': mean_absolute_error})
    return model.predict(val_x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ff', 'lstm', 'tcn', 'transformer'], default='tcn')
    args = vars(parser.parse_args())

    get_prediction(args['model'])

    # TODO print some metric
