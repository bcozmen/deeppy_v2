import torch
import torch.nn as nn

class MaskedTransformerEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.TransformerEncoder(**kwargs)
    def forward(self, x):
        sz = x.shape[1]
        mask = torch.log(torch.tril(torch.ones(sz,sz))).to(x.device)
        return self.encoder(x,mask = mask)