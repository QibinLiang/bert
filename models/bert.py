import torch
from models.embeddings import Embedding
from models.transformer import TransformerBlock

class Bert(torch.nn.Module):
    """ The BERT model(https://arxiv.org/pdf/1810.04805.pdf)

    Args:
        vocab_size (int): the size of the vocabulary
        d_emb (int): the dimension of the embedding
        max_len (int, optional): the maximum length of the input. Defaults to 512.
        n_head (int, optional): the number of heads in the multi-head attention. Defaults to 8.
        n_layers (int, optional): the number of layers in the transformer. Defaults to 6.
        hidden_dim (int, optional): the dimension of the hidden layer in the feed forward network. Defaults to 2048.
        dropout (float, optional): the dropout rate. Defaults to 0.15.

    Output:
        x (torch.Tensor): the output of the BERT model

    """
    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        max_len: int = 125,
        n_head: int = 12,
        n_layers: int = 12,
        hidden_dim: int = 768,
        dropout: int = 0.15
    ) -> None:
    
        super().__init__()
        # bert embedding
        self.embeddings = Embedding(vocab_size, d_emb, max_len, dropout)
        # bert transformers
        self.transformers = torch.nn.ModuleList([
            TransformerBlock(d_emb, hidden_dim, n_head, dropout)
            for _ in range(n_layers)
        ])

    def get_mask(self, x_lens):
        mask = torch.arange(x_lens.max(), device=x_lens.device)[None, :] < x_lens[:, None]
        mask = mask[:, None, None, :]
        return mask

    def forward(self, x, seg, x_lens):
        mask = self.get_mask(x_lens)
        x = self.embeddings(x, seg)
        for transformer in self.transformers:
            x = transformer(x, mask=mask)
        return x

class BertForSequenceClassification(torch.nn.Module):
    """
    The model for sequence classification that predict the senetence

    Args:
        bert (Bert): the BERT model
        vocab_size (int): the size of the vocabulary
        d_emb (int): the dimension of the embedding

    Output:
        is_next (torch.Tensor): log_probability for the next sentence prediction
        token_classify (torch.Tensor): log_probability for the token classification
    """
    def __init__(
        self,
        bert : Bert,
        vocab_size : int, 
        d_emb : int
    ) -> None:
        super().__init__()
        self.bert = bert

        self.is_next_linear = torch.nn.Linear(d_emb, 2)
        self.token_classify_linear = torch.nn.Sequential(
            torch.nn.Linear(d_emb, d_emb),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(d_emb),
            torch.nn.Linear(d_emb, vocab_size),
        )

    def forward(self, x, seg, x_lens):
        x = self.bert(x, seg, x_lens)

        is_next = self.is_next_linear(x[:, 0])
        is_next = torch.nn.functional.log_softmax(is_next, dim=-1)

        token_classify = self.token_classify_linear(x)
        token_classify = torch.nn.functional.log_softmax(token_classify.transpose(-1,-2), dim=-1)

        return is_next, token_classify
        