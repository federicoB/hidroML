import argparse
import keras.backend as K
from src.models.definitions.feed_forward import get_feed_forward_model
from src.models.definitions.lstm import get_lstm_model
from src.models.definitions.temporal_cn import get_tcn_model
from src.models.definitions.transformer import get_transformer_model
from src.data.make_dataset import load_input
from tensorflow.keras.callbacks import EarlyStopping

training_data_ratio = 0.9
sample_length = 128
step_ahead = 3
batch_size = 32


def max_absolute_error(y_true, y_pred):
    return K.max(K.abs(y_pred - y_true), axis=0)[-1]


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=0)[-1]


# TODO make an external callable function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', choices=['ff', 'lstm', 'tcn', 'transformer'], default='tcn')
    parser.add_argument('-e','--epochs', type=int, default=3)
    args = vars(parser.parse_args())

    if args['model'] == 'ff':
        model = get_feed_forward_model(sample_length, step_ahead)
    elif args['model'] == 'lstm':
        model = get_lstm_model(sample_length, step_ahead)
    elif args['model'] == 'tcn':
        model = get_tcn_model(sample_length, step_ahead)
    elif args['model'] == 'transformer':
        model = get_transformer_model(sample_length, step_ahead)

    epochs = args['epochs']

    train_x, train_y, val_x, val_y, val_dates = \
        load_input(sample_length, training_data_ratio, step_ahead)

    model.compile(loss='mse', optimizer='adam', metrics=[max_absolute_error, mean_absolute_error])
    model.build(input_shape=train_x.shape)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.00001, patience=3, restore_best_weights=True)
    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    model.save("models/{}".format(args['model']))
