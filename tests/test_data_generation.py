from set_transformer.data.simulation import gen_max_dataset
import unittest


class TestDataGen(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_max_gen(self):
        dataset = gen_max_dataset(dataset_size=10, set_size=3)
        self.assertEqual(dataset[0].shape[0], 10)
        self.assertEqual(dataset[0].shape[1], 3)
        self.assertEqual(dataset[0].shape[2], 1)


if __name__ == '__main__':
    unittest.main()
