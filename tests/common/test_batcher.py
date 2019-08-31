import unittest
from mleus.common.batcher import Batcher


class TestBatcher(unittest.TestCase):

    def test_batcher(self):
        data = [
            'My name is Ahmed',
            'One doesnt simple write python code',
            'heh',
            'This life journey will be my masterpiece',
            'The unseen blade is the deadliest',
            'Hah!',
            'Make Egypt Great Again',
            '8',
            '9',
            '10'
        ]

        batcher = Batcher(data, batch_size=4)

        self.assertEqual(batcher.total_train_samples, 8)
        self.assertEqual(batcher.total_valid_samples, 1)
        self.assertEqual(batcher.total_test_samples, 1)
        self.assertEqual(batcher.total_train_batches, 2)
        self.assertEqual(batcher.total_valid_batches, 1)
        self.assertEqual(batcher.total_test_batches, 1)

