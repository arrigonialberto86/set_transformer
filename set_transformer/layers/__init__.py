import tensorflow as tf
from tensorflow.keras.layers import Dense


class RFF(tf.keras.layers.Layer):
    """
    Row-wise FeedForward layers.
    """

    def __init__(self, d):
        super(RFF, self).__init__()

        self.linear_1 = Dense(d, activation='relu')
        self.linear_2 = Dense(d, activation='relu')
        self.linear_3 = Dense(d, activation='relu')

    def call(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.linear_3(self.linear_2(self.linear_1(x)))