import torch
import math

class PositionalEmbedding(torch.nn.Module):
    """
    Positional embeddings from "Attention is all you need"(Vaswani et al., 2017)
    (https://arxiv.org/abs/1706.03762)

    Args:
        d_model (int): the dimension of the embedding
        max_len (int, optional): the maximum length of the input. Defaults to 512.

    Output:
        pe (torch.Tensor): the positional embedding
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(torch.nn.Embedding):
    def __init__(self, d_emb):
        # 3 segments: <pad>, <bos>, <eos>
        super().__init__(3, d_emb)

class TokenEmbedding(torch.nn.Embedding):
    def __init__(self, vocab_size, d_emb):
        super().__init__(vocab_size, d_emb)

class Embedding(torch.nn.Module):
    """The BERT embedding that is consist of token embedding, positional 
    embedding, and segment embedding.

    Args:
        vocab_size (int): the size of the vocabulary
        d_emb (int): the dimension of the embedding
        max_len (int, optional): the maximum length of the input. Defaults to 512.
        dropout (float, optional): the dropout rate. Defaults to 0.15.
    
    Output:
        x (torch.Tensor): the embedding of the input
    """
    def __init__(self, vocab_size, d_emb, max_len=512, dropout=0.15):
        super().__init__()
        self.d_emb = d_emb
        self.tok = TokenEmbedding(vocab_size, d_emb)
        self.pos = PositionalEmbedding(d_emb, max_len)
        self.seg = SegmentEmbedding(d_emb)
        self.ln = torch.nn.LayerNorm(d_emb)

    def forward(self, x, seg):
        x = x
        return self.ln(self.tok(x) + self.pos(x) + self.seg(seg))