import unittest
import tensorflow as tf
from set_transformer.layers.attention import scaled_dot_product_attention, MultiHeadAttention


class TestAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_k = tf.constant([[10, 0, 0],
                                   [0, 10, 0],
                                   [0, 0, 10],
                                   [0, 0, 10]], dtype=tf.float32)  # (4, 3)

        self.temp_v = tf.constant([[1, 0],
                                  [10, 0],
                                  [100, 5],
                                  [1000, 6]], dtype=tf.float32)  # (4, 2)

        # This `query` aligns with the second `key`,
        # so the second `value` is returned.
        self.temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

    def test_scaled_dot_product(self):
        # Dimensionality output check
        temp_out, temp_attn = scaled_dot_product_attention(self.temp_q, self.temp_k, self.temp_v, None)
        self.assertEqual(temp_out.shape[0], 1)
        self.assertEqual(temp_out.shape[1], 2)

    def test_multi_head_output_dimension(self):
        temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
        y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
        out = temp_mha(v=y, k=y, q=y)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 60)
        self.assertEqual(out.shape[2], 512)


if __name__ == '__main__':
    unittest.main()
