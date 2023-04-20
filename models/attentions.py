import torch
import math

# TODO :
# 1. change the input of the forward function from (x, mask) to (q, k, v, mask)
# 2. change the output of the forward function from (output) to (output, attention)

class MultiHeadAttention(torch.nn.Module):
    """ MultiHeadAttention module
    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden layer
        num_head (int): the number of heads in the multi-head attention
        dropout (float, optional): the dropout rate. Defaults to 0.15.

    Output:
        output (torch.Tensor): the output of the multi-head attention

    """
    def __init__(self, in_dim, hidden_dim, num_head, dropout=0.15):
        super().__init__()
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // num_head
        self.in_dim = in_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.q_linear = torch.nn.Linear(in_dim, hidden_dim)
        self.k_linear = torch.nn.Linear(in_dim, hidden_dim)
        self.v_linear = torch.nn.Linear(in_dim, hidden_dim)
        self.out_linear = torch.nn.Linear(hidden_dim, in_dim)

    def compute_qkv(self, x):
        batch_size = x.size(0)
        q = self.q_linear(x).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        return q, k, v
    
    def compute_att(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        scores = scores.masked_fill(mask==0, -1e9)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, x, mask=None):
        q, k, v = self.compute_qkv(x)
        output = self.compute_att(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.head_dim * self.num_head)
        output = self.out_linear(output)
        return output

