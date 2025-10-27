import torch
import torch.nn as nn

class TokenizerBeforePosition(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    def forward(self, x, p):
        tokenized_x = self.tokenizer(x)
        return tokenized_x, p
    
class SaneLinearTokenizer(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.linear= nn.Linear(in_features,out_features)
    def forward(self,X):
        x,p = X
        return self.linear(x),p

class ChunkwiseLinearTokenizer(nn.Module):
    def __init__(self, chunk_size, out_dim):
        super().__init__()
        self.chunk_size = chunk_size
        self.out_dim = out_dim
        self.tokenizer = nn.Linear(chunk_size, out_dim)

    def forward(self, x):
        # x: [B, n_chunks * chunk_size]
        D = x.shape[-1]
        orig_shape = tuple(x.shape[:-1])

        n_chunks = D // self.chunk_size
        x = x.view(-1, n_chunks, self.chunk_size)
        tokens = self.tokenizer(x)
        tokens = tokens.reshape(orig_shape + (self.out_dim * n_chunks,))
        return tokens