import unittest
import torch
from top_decoder import TopDecoder


class TestTopDecoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.nhead = 8
        self.tgt_length = 32
        self.mem_length = 64
        self.d_model = 512
        self.num_layers = 5

    def test_forward(self):
        decoder = TopDecoder(self.num_layers, self.d_model, self.nhead)
        mem = torch.rand(self.mem_length, self.batch_size, self.d_model)
        tgt = torch.rand(self.tgt_length, self.batch_size, self.d_model)
        with torch.no_grad():
            y = decoder(tgt, mem)
            self.assertEqual(y.size(), tgt.size())


if __name__ == "__main__":
    unittest.main()
