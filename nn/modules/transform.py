import torch
import torch.nn as nn

class getFirstTokenOutput(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, X):
		X = X[..., 0, :]
		return X
        
class SqueezeLastDimentionInput(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[:-2] + (-1,))
        return x
class SqueezeLastDimention2Inputs(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        z1,z2 = X
        #print(z1.shape)
        #print(z2.shape)
        z1 = z1.view(z1.shape[:-2] + (-1,))
        z2 = z2.view(z2.shape[:-2] + (-1,))
        return torch.cat([z1, z2], dim=-1)

class concatInputs(nn.Module):
    def __init__(self, dim = 0):
        super().__init__()
        self.dim = dim
    def forward(self, X):
        return torch.cat(X, dim=self.dim)