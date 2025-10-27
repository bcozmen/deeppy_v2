#https://arxiv.org/html/2406.09997v1#bib.bib39
#https://github.com/HSG-AIML/SANE
import torch
import torch.nn as nn

from ..base_model import BaseModel
from ...nn.network import Network
from ...nn.optimizer import Optimizer
from ...nn.modules.transform import SqueezeLastDimentionInput, SqueezeLastDimention2Inputs, concatInputs, getFirstTokenOutput
from ...nn.modules.loss import QuaternionLoss, NT_Xent
from ...nn.modules.positional_embedding import SaneLinearTokenizerBeforePosition,SaneXYZPositionalEmbedding
from ...nn.modules.learnable import AttentionPooling


class concatInputsWithPosition(nn.Module):
	def __init__(self, dim = 0, sequence_length = 1, embed_dim=1, num_inputs = 2):
		super().__init__()
		self.dim = dim
		self.sequence_length = sequence_length
		self.embed_dim = embed_dim
		self.num_inputs = num_inputs

		self.unique_pos = nn.Embedding(sequence_length, embed_dim)
		self.layer_pos = nn.Embedding(num_inputs, embed_dim)
		self.rot_token = nn.Embedding(1,embed_dim)

		unique_pos_indices = torch.arange(sequence_length).repeat(num_inputs)
		layer_pos_indices = torch.arange(num_inputs).repeat_interleave(sequence_length)


		self.register_buffer('unique_pos_indices', unique_pos_indices)
		self.register_buffer('layer_pos_indices', layer_pos_indices)
		self._init_weights()

	def _init_weights(self, init_range=0.02):
		nn.init.normal_(self.unique_pos.weight, mean=0.0, std=init_range)
		nn.init.normal_(self.layer_pos.weight, mean=0.0, std=init_range)
		nn.init.normal_(self.rot_token.weight, mean=0.0, std=init_range)
	def forward(self, X):
		p1,p2 = self.unique_pos(self.unique_pos_indices), self.layer_pos(self.layer_pos_indices)
		X = torch.cat(X, dim=self.dim)
		X = X + p1 + p2

		batch_shape = X.shape[:-2]                        # e.g. (batch_size,)
		rot_emb = self.rot_token.weight                       # (1, embed_dim)
		rot_emb = rot_emb.expand(*batch_shape, 1, self.embed_dim)

		output = torch.cat([rot_emb, X], dim=-2)   
		return output

class SaneRotationalHead(nn.Module):
	def __init__(self, latent_dim= 1, proj_dim=1, out_dim=1):
		super().__init__()
		self.proj = nn.Sequential(
			nn.Linear(latent_dim*2, proj_dim),
			nn.ReLU(),
			nn.Linear(proj_dim, out_dim)
		)
		self._init_weights()
	def _init_weights(self):
		for m in self.proj:
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	def forward(self, X):
		proj = self.proj(torch.cat(X, dim = -1))
		return torch.nn.functional.normalize(proj, dim=-1)

class Sane(BaseModel):
	#kwargs = device, criterion
	dependencies = [Network]

	def __init__(self, optimizer_params, max_positions, 
		input_dim= 201, latent_dim = 128, projection_dim = 30, pos_token_size = 10,
		embed_dim=1024, num_heads=4, num_layers=4,  dropout = 0.1, context_size=50, bias = True, 
		gamma = [0.05,0.05], ntx_temp = 0.1, noise_augment = 0):

		super().__init__()

		#Init Loss function
		self.ntx_temp = ntx_temp
		self.rot_crit = QuaternionLoss(loss_type='relative')
		self.recon_crit = nn.MSELoss()
		self.ntx_crit = NT_Xent(temp = ntx_temp)
		

		self.gamma = torch.tensor(gamma).to(self.device)
		self.pos_token_size = pos_token_size
		self.noise_augment = noise_augment

		#Encoder
		self.input_dim = input_dim
		self.max_positions = max_positions
		self.embed_dim = embed_dim
		#Transformerd
		self.context_size = context_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		
		self.dropout = dropout
		self.bias = bias
		self.projection_dim = projection_dim
		self.latent_dim = latent_dim
		self.optimizer_params = optimizer_params
	
		#Some initialization functions are called after the __init__
		#Check base_model.py after_init for more information
	def forward(self, X):
		X,p = X
		z = self.autoencoder.encode((X,p))
		zp = self.project(z[:,:-self.pos_token_size, :])
		y = self.autoencoder.decode((z,p))
		return z, y, zp

	def get_loss(self,X):
		x_1, p_1,m_1,r_1, x_2, p_2,m_2,r_2 = X
		r_1, r_2 = self.rot_crit.euler_to_quaternion(r_1), self.rot_crit.euler_to_quaternion(r_2)

		if self.noise_augment > 0:
			x_1_noisy = x_1 * (1.0 + self.noise_augment * torch.randn_like(x_1))
			x_2_noisy = x_2 * (1.0 + self.noise_augment * torch.randn_like(x_2))
		else:
			x_1_noisy, x_2_noisy = x_1,x_2
		z_1, y_1, zp_1 = self((x_1_noisy, p_1))
		z_2, y_2, zp_2 = self((x_2_noisy, p_2))

		

		q_pred = self.classify((z_1[:,-self.pos_token_size :,:], z_2[:,-self.pos_token_size:,:]))
		
		#Compute reconstruction loss
		x = torch.cat([x_1, x_2], dim=0)
		y = torch.cat([y_1, y_2], dim=0)
		m = torch.cat([m_1, m_2], dim=0)

		recon_loss = self.recon_crit(y*m,x) 

		
		#Compute rotation loss
		rot_loss = self.rot_crit(q_pred, r_1, r_2)

		#Compute NTX loss
		ntx_loss = self.ntx_crit(zp_1, zp_2)
		z_l2_loss = (z_1.pow(2).mean() +  z_2.pow(2).mean()) / 2
		#Compute final loss
		loss_array = torch.stack([recon_loss, ntx_loss, rot_loss, z_l2_loss])
		loss = (loss_array * self.gamma).sum()

		z_distance = (z_1[:,:-self.pos_token_size, :] - z_2[:,:-self.pos_token_size, :]).pow(2)

		loss_array = torch.cat([torch.tensor([loss.item()]), loss_array.detach().cpu()])
		metrics_array = torch.cat([
			z_distance.mean().view(1),
			z_distance.std().view(1)
		])


		self.logger.add("Losses", loss_array )
		self.logger.add("Metrics", metrics_array)
	
		return loss

	def back_propagate(self,loss):
		return self.optimizer.step(loss)

	def encode(self,X):
		return self.autoencoder.encode(X)

	def decode(self,X):
		return self.autoencoder.decode(X)

	def embed(self,X):
		return torch.mean(self.encode(X), dim=1)
	
	# =====================================================================
	#INITIALIZATION FUNCTIONS
	
	def _init_networks(self):
		self.autoencoder, self.autoencoder_params = self.build_autoencoder()
		self.project , self.project_params = self.build_projection_head()
		self.classify, self.classify_params = self.build_classifier_encoder()

		self.nets = [self.autoencoder, self.project, self.classify]
		self.params = [self.autoencoder_params, self.project_params, self.classify_params]
	
	def _init_optimizers(self):
		self.optimizer = self.configure_optimizer()
		self.optimizers = [self.optimizer]
	
	def _init_loss_functions(self):
		self.loss_functions = [self.recon_crit, self.ntx_crit, self.rot_crit]

	def _load_loss_functions(self):
		self.recon_crit, self.ntx_crit, self.rot_crit = self.loss_functions

	def _init_logger(self):
		tags = ["Losses", "Metrics"]
		keys = [["Total", "Recon", "NTX", "Rotation", "Z Norm"], ["Z distance mean", "Z distance std"]]
		self.logger = self.create_logger(tags, keys)

	# =====================================================================
	
	def build_autoencoder(self):
		encoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		decoder = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead= self.num_heads, dim_feedforward = 4* self.embed_dim, batch_first= True, norm_first = True, dropout=self.dropout, bias= self.bias, activation = nn.GELU())
		
		blocks = [SaneLinearTokenizerBeforePosition,SaneXYZPositionalEmbedding, nn.Dropout, nn.TransformerEncoder, nn.Linear]
		encoder_params = {
			"blocks": blocks,
			"block_args":[
				{
					"in_features": self.input_dim,
					"out_features" : self.embed_dim,
				},
				{
					"max_positions" : self.max_positions,
					"embed_dim" : self.embed_dim,
					"input_dim" : self.input_dim
				},
				{
					"p" : self.dropout
				},
				{
					"encoder_layer": encoder,
					"num_layers":self.num_layers,
				},
				{
					"in_features" : self.embed_dim,
					"out_features":self.latent_dim,
				}
			],
		}
		decoder_params = {
			"blocks": blocks,
			"block_args":[
				{
					"in_features": self.latent_dim,
					"out_features" : self.embed_dim,
				},
				{
					"max_positions" : self.max_positions,
					"embed_dim" : self.embed_dim,
					"input_dim" : self.input_dim
				},
				{
					"p" : self.dropout
				},
				{
					"encoder_layer":decoder,
					"num_layers":self.num_layers,
				},
				{
					"in_features" : self.embed_dim,
					"out_features":self.input_dim,
				}
			],
		}

		network_params = {
			"arch_params": encoder_params,
			"decoder_params" : decoder_params,
			"task" : "autoencoder",
		}	

		return Network(**network_params).to(self.device), network_params
	def build_projection_head(self):
		arch_params1 = {
			"blocks":[AttentionPooling],
			"block_args" : [{"latent_dim" : self.latent_dim}]
		}

		arch_params2 = {
			"layers" : [self.latent_dim, self.projection_dim, self.projection_dim//2],
			"blocks":[nn.Linear, nn.ReLU],
			"out_act" : nn.Identity
		}

		network_params = {
			"arch_params": [arch_params1, arch_params2],
		}
		return Network(**network_params).to(self.device), network_params

	
	def build_classifier_encoder(self):
		encoder_params = {
			"d_model" : self.latent_dim, 
			"nhead"  :  2, 
			"dim_feedforward" : 2* self.latent_dim, 
			"batch_first" :  True,
			"norm_first" : True, 
			"dropout" : self.dropout, 
			"activation" : nn.GELU(),
		}

		arch_params = {
			"blocks":[concatInputsWithPosition, nn.TransformerEncoderLayer, getFirstTokenOutput, nn.Linear],
			"block_args" : [
				{
					"dim":1,
					"sequence_length" : self.pos_token_size,
					"embed_dim" : self.latent_dim,
				},
				encoder_params,
				{},
				{
					"in_features" : self.latent_dim,
					"out_features" : 4,
				},
			]
		}
		network_params = {
			"arch_params": [arch_params],
		}

		return Network(**network_params).to(self.device), network_params

	def build_classifier(self):
		arch_params1 = {
			"blocks":[SaneRotationalHead],
			"block_args" : [
				   {
					"latent_dim" : self.latent_dim,
					"proj_dim" : self.projection_dim,
					"out_dim" : 4
					}
				   ]
		}
		network_params = {
			"arch_params": [arch_params1],
		}

		return Network(**network_params).to(self.device), network_params
	def configure_optimizer(self):
		decay_params = []
		nodecay_params = []

		for net in self.nets:
			for param in net.model.parameters():
				if not param.requires_grad:
					continue
				if param.dim() >= 2:
					decay_params.append(param)
				else:
					nodecay_params.append(param)

		optim_groups = [
			{"params": decay_params, "weight_decay": self.optimizer_params["optimizer_args"]["weight_decay"]},
			{"params": nodecay_params, "weight_decay": 0.0},
		]

		del self.optimizer_params["optimizer_args"]["weight_decay"]
		return Optimizer(optim_groups, **self.optimizer_params)
	

	# =====================================================================
	#HELPER FUNCTIONS
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(   module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def load(self, file_name):
		#Load the model from the class.
		#First initialize a new object, and then load the checkpoint
		if isinstance(file_name, dict):
			checkpoint = file_name
		else:
			checkpoint = torch.load(file_name + "/checkpoint.pt", weights_only = False)
		
		dicts = checkpoint["nets"]
		objs = checkpoint["objs"]
		optimizer_dicts = checkpoint["optimizer"]
		

		for net,net_dicts in zip(self.nets, dicts):
			net.load_states(net_dicts)
		if optimizer_dicts is not None:
			[optimizer.load_states(dic) for optimizer, dic in zip(self.optimizers, optimizer_dicts)]

		self.loss_functions = objs
		self._load_loss_functions()
		return self