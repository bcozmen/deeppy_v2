import torch
import torch.nn as nn

class TokenizerBeforePosition(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    def forward(self, x, p):
        tokenized_x = self.tokenizer(x)
        return tokenized_x, p
    
class LinearTokenizerBeforePosition(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.linear= nn.Linear(in_features,out_features)
    def forward(self,X):
        x,p = X
        return self.linear(x),p

class SaneLinearTokenizerBeforePosition(nn.Module):
    def __init__(self,in_features, out_features, max_positions = [50000, 25]):
        super().__init__()
        self.divide_ix = max_positions[1]
        self.linear_hash= nn.Linear(in_features,out_features)
        self.linear_mlp = nn.Linear(in_features,out_features)
        self._init_weights()
    def forward(self,X):
        x,p = X
        hash,mlp = x[:,:-self.divide_ix],  x[:, -self.divide_ix:]
        return torch.cat( (self.linear_hash(hash), self.linear_mlp(mlp)), dim=1),p
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.linear_hash.weight)
        nn.init.kaiming_uniform_(self.linear_mlp.weight)
        if self.linear_hash.bias is not None:
            nn.init.zeros_(self.linear_hash.bias)
        
        if self.linear_mlp.bias is not None:
            nn.init.zeros_(self.linear_mlp.bias)

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

class OrderedPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings,embedding_dim)

    def forward(self,x):
        t = x.shape[1]
        pos = torch.arange(0, t, dtype=torch.long, device = x.device).unsqueeze(0) 
        return x + self.embed(pos)


class ChunkwisePositionalEmbedding(nn.Module):
    def __init__(self, max_positions = 5, embed_dim = 128, chunk_size = 2):
        super().__init__()
        self.max_positions = max_positions
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        
        self.positional_embedding = nn.Embedding(max_positions, embed_dim)

        self._init_weights()
    def forward(self, p):
        D = p.shape[-1]
        orig_shape = tuple(p.shape[:-1])

        n_chunks = D // self.chunk_size
        p = p.view(-1, n_chunks, self.chunk_size)
        pe = self.positional_embedding(p)
        pe = pe.reshape(orig_shape + (self.embed_dim * n_chunks,))

        return pe
    def _init_weights(self):
        nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)

class SaneXYZPositionalEmbedding(nn.Module):
    def __init__(self, max_positions = [5000000, 25], embed_dim = 128, input_dim = 256):
        super().__init__()
        self.max_positions = max_positions
        self.embed_dim = embed_dim

        self.hash_xyz = nn.Linear(3, embed_dim)
        self.hash_index_global = ChunkwisePositionalEmbedding(max_positions=max_positions[0], embed_dim= 2*embed_dim//input_dim, chunk_size=1)
        self.hash_layer = ChunkwisePositionalEmbedding(max_positions=16, embed_dim= 2*embed_dim//input_dim, chunk_size=1)
        self.hash_index_layerwise = ChunkwisePositionalEmbedding(max_positions=700000, embed_dim= 2*embed_dim//input_dim, chunk_size=1)

        self.mlp_embed = nn.Embedding(max_positions[1], embed_dim)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.hash_xyz.weight)
        if self.hash_xyz.bias is not None:
            nn.init.zeros_(self.hash_xyz.bias)
        
        nn.init.normal_(self.mlp_embed.weight, mean=0.0, std=0.02)
        
    def forward(self, X):
        x,p = X

        hash, mlp = p[:,:-self.max_positions[1]], p[:, -self.max_positions[1]:].long()
        hash_xyz, hash_indices = hash[:,:,:3], hash[:,:,3:].long()

        l = hash_indices.shape[-1] // 3
        
        hash_indices_global, hash_layers, hash_indices_layerwise = self.hash_index_global(hash_indices[...,:l]), self.hash_layer(hash_indices[...,l:2*l]), self.hash_index_layerwise(hash_indices[...,2*l:])
        he_xyz, mlpe = self.hash_xyz(hash_xyz), self.mlp_embed(mlp[...,0])

 
        he = he_xyz + hash_indices_global + hash_layers + hash_indices_layerwise

        he = torch.cat((he, mlpe), dim=1)

        return x + he

class SaneXYZPositionalEmbedding_Linear(nn.Module):
    def __init__(self, max_positions = [25], embed_dim = 128):
        super().__init__()
        self.max_positions = max_positions
        self.embed_dim = embed_dim

        self.hash_embed = nn.Linear(3, embed_dim)
    
    def forward(self, X):
        x,p = X
        return x + self.hash_embed(p)
class SanePositionalEmbedding(nn.Module):
    def __init__(self, max_positions=[48, 256], embed_dim=128):
        super().__init__()
        self.max_positions = max_positions
        self.embed_dim = embed_dim
        if len(max_positions) == 2:
            self.pe1 = nn.Embedding(max_positions[0], embed_dim // 2)
            self.pe2 = nn.Embedding(max_positions[1], embed_dim // 2)
            self.pe3 = None
        elif len(max_positions) == 3:
            self.pe1 = nn.Embedding(max_positions[0], embed_dim // 2)  # add 1 + 2
            self.pe2 = nn.Embedding(max_positions[1], embed_dim // 2)  # add 1 + 2
            self.pe3 = nn.Embedding(max_positions[2], embed_dim // 2)  # cat 1+2 & 3

    def forward(self, X):
        inputs, pos = X
        pos_emb1 = self.pe1(pos[..., 0])
        pos_emb2 = self.pe2(pos[..., 1])
        if self.pe3 is not None:
            pos_emb3 = self.pe3(pos[:, :, 2])
            pos_emb = [pos_emb1 + pos_emb2, pos_emb3]
        else:
            pos_emb = [pos_emb1, pos_emb2]

        pos_emb = torch.cat(pos_emb, dim=2)

        out = inputs + pos_emb
        return out


class FrequencyPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x


