import unittest
import torch
from top_decoder import TopDecoder


class TestTopDecoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.nhead = 8
        self.tgt_length = 32
        self.enc_length = 64
        self.d_model = 512
        self.num_layers = 5

    def test_forward(self):
        decoder = TopDecoder(self.num_layers, self.d_model, self.nhead)
        memory = torch.rand(self.enc_length, self.batch_size, self.d_model)
        x = torch.rand(self.tgt_length, self.batch_size, self.d_model)
        with torch.no_grad():
            y = decoder(x, memory)
            self.assertEqual(y.size(), x.size())


if __name__ == "__main__":
    unittest.main()
