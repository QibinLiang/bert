import torch
from models.attentions import MultiHeadAttention

# TODO :
# 1. update input and output of the attention layer.

class TransformerBlock(torch.nn.Module):
    """ The transformer encoder block(https://arxiv.org/abs/1706.03762)

    [input]----{MultiHeadAttention}----{Add & Norm}---->[output_att]---->{FeedForward}----{Add & Norm}------>[output]
       |                                    ^                |                                 ^
       |                                    |                |                                 |
       |---------skipconnection-------------|                |----------skipconnection---------|


    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden layer in the feed forward network
        n_head (int): the number of heads in the multi-head attention
        dropout (float, optional): the dropout rate. Defaults to 0.15.
        dense_layer (bool, optional): whether to use the dense layer. Defaults to True.
    
    Output:
        x (torch.Tensor): the output of the transformer block
    """
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        n_head: int, 
        dropout: int = 0.15, 
    ) -> None:
        super().__init__()
        ffn_dim = hidden_dim * 4

        self.attention = MultiHeadAttention(in_dim, hidden_dim, n_head, dropout)
        self.ffn = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, ffn_dim),
                torch.nn.GELU(),
                torch.nn.Linear(ffn_dim, hidden_dim),
            )
        self.dropoutAtt = torch.nn.Dropout(dropout)
        self.dropoutFFN = torch.nn.Dropout(dropout)
        self.normAtt = torch.nn.LayerNorm(hidden_dim)
        self.normFFN = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        x = self.normAtt(x + self.dropoutAtt(self.attention(x, mask)))
        x = self.normFFN(x + self.dropoutFFN(self.ffn(x)))
        return x

class SkiplessTrainsformerBlock(torch.nn.Module):
    pass