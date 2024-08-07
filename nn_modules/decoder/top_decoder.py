from torch import nn
from torch import Tensor


class TopDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int) -> None:
        super(TopDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)])

    def forward(self, tgt: Tensor, mem: Tensor) -> Tensor:
        """Pass the inputs through the decoder layer.

        Parameters
        ----------
        tgt : Tensors
            [tgt_length, batch_size, d_model]
            the sequence to the decoder layer (required).
        mem : Tensor,
            [mem_length, batch_size, d_model]
            the sequence from the last layer of the encoder (required).

        Returns
        -------
        Tensor
            [batch_size, tgt_length, d_model]
        """
        for layer in self.layers:
            tgt = layer(tgt, mem)
        return tgt
