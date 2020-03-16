import numpy as np
import tensorflow as tf


def gen_max_dataset(dataset_size=100000, set_size=9, seed=0):
    """
    The number of objects per set is constant in this toy example
    """
    np.random.seed(seed)
    x = np.random.uniform(1, 100, (dataset_size, set_size))
    y = np.max(x, axis=1)
    x, y = np.expand_dims(x, axis=2), np.expand_dims(y, axis=1)
    return tf.cast(x, 'float32'), tf.cast(y, 'float32')
