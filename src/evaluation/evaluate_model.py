import argparse
from src.evaluation.get_flops import get_flops
from src.data.make_dataset import load_input
from src.models.train_model import batch_size
from src.models.train_model import sample_length, training_data_ratio, step_ahead, \
    max_absolute_error, mean_absolute_error
import tensorflow as tf

def evaluate_model(model_name):

    n_flops = get_flops(model_name)

    train_x, train_y, val_x, val_y, val_dates = \
        load_input(sample_length, training_data_ratio, step_ahead)
    # Note: get_flops() already load the model but we can't use it here, we need to load it again
    model = tf.keras.models.load_model("models/{}".format(model_name),
                                       custom_objects={'max_absolute_error': max_absolute_error,
                                                       'mean_absolute_error': mean_absolute_error})
    n_params = model.count_params()
    results = model.evaluate(val_x, val_y, batch_size=batch_size)

    metric = results[2]
    return n_params, n_flops, metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ff', 'lstm', 'tcn', 'transformer'], default='tcn')
    args = vars(parser.parse_args())

    n_params, n_flops, metric = evaluate_model(args['model'])
    output = "{} parameters {} flops {} metric".format(n_params,n_flops,metric)
    print(output)