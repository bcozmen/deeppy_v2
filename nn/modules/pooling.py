
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AttentionPooling"]


class AttentionPooling(nn.Module):
	"""Attention-based pooling with numeric safeguards.

	Changes vs original:
	- Use explicit matmul and sum instead of some transposes/bmm shapes.
	- Compute softmax in float32 when input is float16 to avoid fp16 instability.
	- Avoid in-place ops and keep shapes explicit.
	"""

	def __init__(self, latent_dim: int):
		super().__init__()
		self.latent_dim = latent_dim
		# small init scale
		self.query = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
		# scaling factor for dot-product attention
		self.scale = math.sqrt(latent_dim)

	def forward(self, z: torch.Tensor) -> torch.Tensor:
		"""z: (batch_size, context, latent_dim)

		returns: pooled (batch_size, latent_dim)
		"""
		batch_size, context, latent_dim = z.shape

		# expand query to batch: (batch, 1, latent_dim)
		q = self.query.unsqueeze(0).expand(batch_size, -1, -1)

		# Compute attention in FP32 for AMP stability
		z_fp32 = z.float()
		q_fp32 = q.float()
		
		# logits: (batch, context)  -- compute by matmul and squeeze
		# (batch, context, latent) @ (batch, latent, 1) -> (batch, context, 1)
		logits = torch.matmul(z_fp32, q_fp32.transpose(1, 2)).squeeze(-1) / self.scale

		# numeric stabilization
		logits = logits - logits.max(dim=1, keepdim=True)[0]

		attn = F.softmax(logits, dim=1)
		
		# Safe entropy computation with clamping
		attn_clamped = torch.clamp(attn, min=1e-6)
		entropy = -(attn * torch.log(attn_clamped)).sum(dim=1).mean()
		normalized_entropy = entropy / math.log(attn.shape[1])
		
		# apply attention: (batch, context, 1) * (batch, context, latent) -> sum over context
		pooled = (attn.unsqueeze(-1) * z_fp32).sum(dim=1)

		return pooled.to(z.dtype), normalized_entropy


class MultiQueryAttentionPooling(nn.Module):
	"""
	Attention pooling with multiple learned queries.
	
	Input:  (batch, context, latent_dim)
	Output: (batch, num_queries, latent_dim)
	"""
	def __init__(self, latent_dim: int, num_queries: int):
		super().__init__()
		self.latent_dim = latent_dim
		self.num_queries = num_queries
		self.queries = nn.Parameter(torch.randn(num_queries, latent_dim) * 0.02)
		self.scale = math.sqrt(latent_dim)

	def forward(self, z):
		b, c, d = z.shape

		q = self.queries.unsqueeze(0).expand(b, -1, -1)

		# Always compute attention in fp32 for AMP stability
		z_fp32 = z.float()
		q_fp32 = q.float()
		logits = torch.matmul(q_fp32, z_fp32.transpose(1, 2)) / self.scale

		# softmax stabilization
		logits = logits - logits.max(dim=-1, keepdim=True)[0]
		attn = F.softmax(logits, dim=-1)

		# entropy computation with safe clamping
		attn_clamped = torch.clamp(attn, min=1e-6)
		entropy = -(attn * torch.log(attn_clamped)).sum(dim=-1).mean()
		normalized_entropy = entropy / math.log(attn.shape[-1])

		# pooling in fp32, then convert back
		pooled = torch.matmul(attn, z_fp32) # (b, num_queries, latent_dim)
		return pooled.to(z.dtype), normalized_entropy