import unittest
import torch

from top_encoder import TopEncoder

class TestTopEncoder(unittest.TestCase):
    def setUp(self):
        """Construct an encoder block according to specified parameters."""
        self.num_layers = 6
        self.d_model = 512
        self.nhead = 8
        self.batch_size = 32
        self.seq_len = 10
        self.top_encoder = TopEncoder(self.num_layers, self.d_model, self.nhead)
    
    def test_num_layers(self):
        """Check if the number of layers constructed is the same as specified."""
        self.assertEqual(len(self.top_encoder.layers), self.num_layers)

    def test_output_shape(self):
        """Check if the output has the same dimension as the input."""
        x = torch.rand(self.seq_len, self.batch_size, self.d_model)
        z = self.top_encoder.forward(x)
        self.assertEqual(z.size(dim=0), self.seq_len)
        self.assertEqual(z.size(dim=1), self.batch_size)
        self.assertEqual(z.size(dim=2), self.d_model)


if __name__ == '__main__':
    unittest.main()
