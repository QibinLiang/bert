import torch 

# TODO :
# 1. update the forward function for the transformer module
# 2. decouple the transformer module from the ResBlock module

# Residual Block
class ResBlock(torch.nn.Module):
    def __init__(self, layer, out_dim, dropout=0.15):
        super().__init__()
        self.layer = layer
        self.norm = torch.nn.LayerNorm(out_dim)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # if mask is not None:
        #     return self.norm(x + self.layer(x, x, x, mask)[0])
        if mask is not None:
            return self.norm(x + self.dropout(self.layer(x, mask)))
        return self.norm(x + self.dropout(self.layer(x)))