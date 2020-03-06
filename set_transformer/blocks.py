# Referencing https://arxiv.org/pdf/1810.00825.pdf
# and the original PyTorch implementation https://github.com/TropComplique/set-transformer/blob/master/blocks.py
from tensorflow import repeat
from tensorflow.keras.layers import LayerNormalization, Dense
import tensorflow as tf
from set_transformer.layers.attention import MultiHeadAttention
from set_transformer.layers import RFF


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d: int, h: int, rff: RFF):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead = MultiHeadAttention(d, h)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6, dtype='float32')
        self.layer_norm2 = LayerNormalization(epsilon=1e-6, dtype='float32')
        self.rff = rff

    def call(self, x, y):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class SetAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d: int, h: int, rff: RFF):
        super(SetAttentionBlock, self).__init__()
        self.mab = MultiHeadAttentionBlock(d, h, rff)

    def call(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


class InducedSetAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d: int, m: int, h: int, rff1: RFF, rff2: RFF):
        """
        Arguments:
            d: an integer, input dimension.
            m: an integer, number of inducing points.
            h: an integer, number of heads.
            rff1, rff2: modules, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super(InducedSetAttentionBlock, self).__init__()
        self.mab1 = MultiHeadAttentionBlock(d, h, rff1)
        self.mab2 = MultiHeadAttentionBlock(d, h, rff2)
        self.inducing_points = tf.random.normal(shape=(1, m, d))

    def call(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        b = tf.shape(x)[0]
        p = self.inducing_points
        p = repeat(p, (b), axis=0)  # shape [b, m, d]

        h = self.mab1(p, x)  # shape [b, m, d]
        return self.mab2(x, h)


class PoolingMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d: int, k: int, h: int, rff: RFF, rff_s: RFF):
        """
        Arguments:
            d: an integer, input dimension.
            k: an integer, number of seed vectors.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super(PoolingMultiHeadAttention, self).__init__()
        self.mab = MultiHeadAttentionBlock(d, h, rff)
        self.seed_vectors = tf.random.normal(shape=(1, k, d))
        self.rff_s = rff_s

    @tf.function
    def call(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d]
        """
        b = tf.shape(z)[0]
        s = self.seed_vectors
        s = repeat(s, (b), axis=0)  # shape [b, k, d]
        return self.mab(s, self.rff_s(z))


class STEncoder(tf.keras.layers.Layer):
    def __init__(self, d=12, m=6, h=6):
        super(STEncoder, self).__init__()

        # Embedding part
        self.linear_1 = Dense(d, activation='relu')

        # Encoding part
        self.isab_1 = InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        self.isab_2 = InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))

    def call(self, x):
        return self.isab_2(self.isab_1(self.linear_1(x)))


class STDecoder(tf.keras.layers.Layer):
    def __init__(self, out_dim, d=12, h=2, k=8):
        super(STDecoder, self).__init__()

        self.PMA = PoolingMultiHeadAttention(d, k, h, RFF(d), RFF(d))
        self.SAB = SetAttentionBlock(d, h, RFF(d))
        self.output_mapper = Dense(out_dim)
        self.k, self.d = k, d

    def call(self, x):
        decoded_vec = self.SAB(self.PMA(x))
        decoded_vec = tf.reshape(decoded_vec, [-1, self.k * self.d])
        return tf.reshape(self.output_mapper(decoded_vec), (tf.shape(decoded_vec)[0],))
