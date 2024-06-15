import torch
import torch.nn as nn

class TopEncoder(nn.Module):
    """The encoder is a block for extracting features from input sequences."""
    def __init__(self, num_layers: int, d_model: int, nhead: int):
        """Initialize the encoder block as a stack of N identical layers.

        Args:
            num_layers: The number of identical layers, N.
            d_model: The dimension of output feature representation.
            nhead: The number of attention heads.
        """
        super(TopEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])

    def forward(self, x):
        """Receives an input sequence and builds a feature representation of it.

        Args:
            x: The input sequence.
        """
        for layer in self.layers:
            x = layer(x)
        return x
