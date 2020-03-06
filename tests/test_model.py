import unittest
from set_transformer.model import BasicSetTransformer
from set_transformer.data.simulation import gen_max_dataset
import tensorflow as tf


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        # Integration test
        self.X, self.y = gen_max_dataset(300, 3)

    def test_basic_transfomer(self):
        set_transformer = BasicSetTransformer()
        set_transformer.compile(loss='mae', optimizer='adam')
        set_transformer.fit(self.X, self.y, epochs=1)
        prediction = set_transformer.predict(tf.expand_dims(self.X[0], axis=0))
        self.assertEqual(prediction.shape[0], 1)


if __name__ == '__main__':
    unittest.main()