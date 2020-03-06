import unittest
import tensorflow as tf
from set_transformer.blocks import MultiHeadAttentionBlock, SetAttentionBlock, InducedSetAttentionBlock, \
    PoolingMultiHeadAttention

from set_transformer.layers import RFF


class TestBlocks(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_multi_attention_block_dim(self):
        x_data = tf.random.normal(shape=(10, 2, 9))
        y_data = tf.random.normal(shape=(10, 3, 9))
        rff = RFF(d=9)
        mab = MultiHeadAttentionBlock(9, 3, rff=rff)
        mab_output = mab(x_data, y_data)
        self.assertEqual(mab_output.shape[0], 10)
        self.assertEqual(mab_output.shape[1], 2)
        self.assertEqual(mab_output.shape[2], 9)

    def test_set_attention_block_dim(self):
        x_data = tf.random.normal(shape=(10, 2, 9))
        rff = RFF(d=9)
        sab = SetAttentionBlock(9, 3, rff=rff)
        sab_output = sab(x_data)
        self.assertEqual(sab_output.shape[0], 10)
        self.assertEqual(sab_output.shape[1], 2)
        self.assertEqual(sab_output.shape[2], 9)

    def test_induced_set_attention_block_dim(self):
        z = tf.random.normal(shape=(10, 2, 9))
        rff, rff_s = RFF(d=9), RFF(d=9)
        pma = InducedSetAttentionBlock(d=9, m=10, h=3, rff1=rff, rff2=rff_s)
        output = pma(z)
        self.assertEqual(output.shape[0], 10)
        self.assertEqual(output.shape[1], 2)
        self.assertEqual(output.shape[2], 9)

    def test_pma_block_dim(self):
        z = tf.random.normal(shape=(10, 2, 9))
        rff, rff_s = RFF(d=9), RFF(d=9)
        pma = PoolingMultiHeadAttention(d=9, k=10, h=3, rff=rff, rff_s=rff_s)
        output = pma(z)
        self.assertEqual(output.shape[0], 10)
        self.assertEqual(output.shape[1], 10)
        self.assertEqual(output.shape[2], 9)


if __name__ == '__main__':
    unittest.main()