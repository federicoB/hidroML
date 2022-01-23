import tensorflow as tf
from src.models.train_model import sample_length, training_data_ratio, step_ahead, \
    max_absolute_error, mean_absolute_error


# From https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras

def get_flops(model_name: str):
        session = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()
        with graph.as_default():
            with session.as_default():
                # load the model inside a graph session, in this way the flops will be counted
                tf.keras.models.load_model("models/{}".format(model_name),
                                           custom_objects={'max_absolute_error': max_absolute_error,
                                                           'mean_absolute_error': mean_absolute_error})

                run_meta = tf.compat.v1.RunMetadata()
                # Or, build your own options:

                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()


                flops = tf.compat.v1.profiler.profile(graph=graph,
                run_meta=run_meta, cmd='op', options=opts)

        tf.compat.v1.reset_default_graph()


        return flops.total_float_ops
