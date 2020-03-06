import unittest
import tensorflow as tf
from set_transformer.layers import RFF


class TestLayers(unittest.TestCase):
    def test_rff(self):
        mlp = RFF(3)
        y = mlp(tf.ones(shape=(2, 4, 3)))
        self.assertEqual(len(mlp.weights), 6)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 4)
        self.assertEqual(y.shape[2], 3)


if __name__ == '__main__':
    unittest.main()
