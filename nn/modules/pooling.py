import torch
import torch.nn as nn

__all__ = ["AttentionPooling"]

class AttentionPooling(nn.Module):
	def __init__(self, latent_dim):
		super().__init__()
		self.query = nn.Parameter(torch.randn(1, latent_dim) * 0.02) 
	def forward(self, z):
		# z: (batch_size, context, latent_dim)
		batch_size, context, latent_dim = z.shape
		q = self.query.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)  # (batch_size, 1, latent_dim)
		attn_scores = torch.bmm(z, q) / (latent_dim ** 0.5) # (batch_size, context, 1)
		attn_weights = torch.softmax(attn_scores, dim=1)  # softmax over context dim
		pooled = torch.bmm(attn_weights.transpose(1, 2), z).squeeze(1)  # (batch_size, latent_dim)
		# Project and normalize
		return pooled