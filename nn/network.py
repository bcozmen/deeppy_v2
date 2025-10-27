import torch
import torch.nn as nn
from ..env_config import env_config


class Network(nn.Module):
	"""
	A flexible neural network wrapper class for PyTorch.
	
	This class provides a high-level interface for creating different types of neural networks
	including regression, classification, and autoencoder architectures. It handles automatic
	layer generation, torch.compile optimization, and task-specific forward pass logic.
	
	Features:
	- Supports multiple task types (regression, classification, autoencoder)
	- Automatic layer generation with flexible architecture specification
	- Optional torch.compile for performance optimization
	- Built-in model state saving/loading
	- Intelligent handling of different layer types (normalization, activation, etc.)
	
	Attributes
	----------
	task : str
		Network task type. One of ["reg", "autoencoder", "classify"]
		- "reg": Standard regression/general purpose network
		- "classify": Classification with threshold-based output during inference
		- "autoencoder": Encoder-decoder architecture with separate encode/decode methods
	classify_threshold : float
		Threshold for binary classification output (default: 0.5)
		During inference, returns (logits > threshold).float()
	arch_params : list or dict
		Architecture parameters for network generation
	decoder_params : list or dict, optional
		Decoder architecture parameters (required for autoencoder task)
	model : nn.Sequential
		The main network model
	encode : nn.Sequential, optional
		Encoder part of autoencoder (only for autoencoder task)
	decode : nn.Sequential, optional
		Decoder part of autoencoder (only for autoencoder task)
	encoder_len : int
		Number of layers in encoder (for autoencoder task)
	"""
	def __init__(self, arch_params, decoder_params=None, task="reg", classify_threshold=0.5, weight_init="uniform"):
		"""
		Initialize the Network.
		
		Args:
			arch_params (dict or list): Architecture specification for the main network.
				If dict, will be converted to a single-element list.
			decoder_params (dict or list, optional): Architecture specification for the decoder
				(required only for autoencoder task).
			task (str): Network task type. One of ["reg", "autoencoder", "classify"].
				- "reg": Standard regression/general purpose network
				- "classify": Classification with threshold-based output during inference
				- "autoencoder": Encoder-decoder architecture with separate encode/decode methods
			classify_threshold (float): Threshold for classification output during inference.
		"""
		super(Network, self).__init__()

		# Define known activation and layer types for internal processing
		self.activation_names = nn.modules.activation.__all__ + ['Identity']
		self.layer_applied_names = [
			"Bilinear", "Conv1d", "Conv2d", "Conv3d", 
			"ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Linear"
		]

		# Store configuration
		self.torch_compile = env_config.torch_compile
		self.task = task
		self.classify_threshold = classify_threshold
		self.weight_init = weight_init

		# Ensure parameters are in list format for consistent processing
		if isinstance(arch_params, dict):
			arch_params = [arch_params]
		if isinstance(decoder_params, dict):
			decoder_params = [decoder_params]

		# Validate task type
		if task not in ["reg", "autoencoder", "classify"]:
			raise ValueError(f"Invalid task '{task}'. Must be one of ['reg', 'autoencoder', 'classify']")
		if task == "autoencoder" and decoder_params is None:
			raise ValueError("Decoder parameters must be provided for autoencoder task")

		# Store parameters and initialize model components
		self.arch_params = arch_params
		self.decoder_params = decoder_params
		self.encode, self.decode, self.encoder_len = None, None, 0
		
		# Generate the model architecture
		self._generate_model()

		
	def forward(self, X):
		"""
		Forward pass through the network.
		
		Args:
			X (torch.Tensor): Input tensor
			
		Returns:
			torch.Tensor: Network output
				- For autoencoder: Reconstructed input
				- For classification (inference): Binary output based on threshold
				- For classification (training) or regression: Raw logits
		"""
		if self.task == "autoencoder":
			return self.decode(self.encode(X))
		
		logits = self.model(X)
		
		# Apply threshold-based classification during inference
		if not self.training and self.task == "classify":
			return (logits > self.classify_threshold).float()
		
		return logits

	def save_states(self):
		"""
		Save the current state of the network.
		
		Returns:
			dict: Dictionary containing the model's state_dict
		"""
		return {
			"net": self.model.state_dict(),
		}

	def load_states(self, state_dict):
		"""
		Load a previously saved network state.
		
		Args:
			state_dict (dict): Dictionary containing the model's state_dict
		"""
		self.model.load_state_dict(state_dict["net"])

	def _compile(self, model):
		"""
		Apply torch.compile optimization if enabled.
		
		Args:
			model (nn.Module): Model to potentially compile
			
		Returns:
			nn.Module: Compiled or original model
		"""
		if self.torch_compile:
			return torch.compile(model)
		return model

	def _generate_model(self):
		"""
		Generate the complete network architecture.
		
		Creates the network based on arch_params and decoder_params.
		For autoencoder tasks, splits the model into encoder and decoder parts.
		Applies torch.compile optimization if enabled.
		"""
		net = []

		# Generate main architecture layers
		for param in self.arch_params:
			net += self._generate_layers(**param)

		# For autoencoders, add decoder layers after encoder
		if self.task == "autoencoder":
			self.encoder_len = len(net)
			for param in self.decoder_params:
				net += self._generate_layers(**param)
		
		# Create the sequential model
		self.model = nn.Sequential(*net)

		# Set up encoder/decoder for autoencoders or compile full model
		if self.task == "autoencoder":
			self.encode = self._compile(self.model[:self.encoder_len])
			self.decode = self._compile(self.model[self.encoder_len:])
		elif self.torch_compile:
			self.model = self._compile(self.model)

	def _generate_layers(self, layers=[None, None], blocks=[], block_args=[], 
						out_act=nn.Identity, out_params={}, weight_init="uniform"):
		"""
		Generate multiple layers with different input and output sizes.
		
		This method creates a sequence of blocks with incrementally changing dimensions.
		It's designed to handle multi-layer architectures where each layer has different
		input/output dimensions.
		
		Args:
			layers (list): List defining the dimension progression (e.g., [128, 64, 32]).
				Must have at least 2 elements to define input->output transitions.
			blocks (list): List of nn.Module classes to use for each layer block.
			block_args (list): List of dictionaries containing arguments for each block.
			out_act (nn.Module): Final activation function for the last layer.
			out_params (dict): Parameters for the output activation function.
			weight_init (callable, optional): Weight initialization function (currently unused).
			
		Returns:
			list: List of instantiated layers ready for nn.Sequential
			
		Raises:
			ValueError: If out_act is None when layers are provided.
		"""
		if out_act is None and len(layers) > 0:
			raise ValueError("out_act cannot be None. Please use nn.Identity")
		
		net = []
		
		# Ensure block_args has the same length as blocks by padding with empty dicts
		block_args = block_args + [{} for _ in range(len(blocks) - len(block_args))]

		# Generate blocks for each layer transition
		this_out_act, this_out_params = None, {}
		for ix, (inp, out) in enumerate(zip(layers[:-1], layers[1:])):
			# Apply output activation only to the last layer
			if ix == len(layers) - 2:
				this_out_act = out_act
				this_out_params = out_params
			
			# Generate the block with given input and output sizes
			net.extend(self._generate_block(blocks, block_args, inp, out, 
										  out_act=this_out_act, out_params=this_out_params))
		

		return net

	def _generate_block(self, blocks, block_args, inp=None, out=None, out_act=None, out_params={}):
		"""
		Generate a single block with given input and output sizes.
		
		This method creates a sequence of neural network layers from a list of layer classes
		and their corresponding arguments. It handles special cases for layers that require
		input/output dimensions and normalization layers.
		
		Args:
			blocks (list): List of nn.Module classes to instantiate
			block_args (list): List of dictionaries containing arguments for each layer
			inp (int, optional): Input dimension size
			out (int, optional): Output dimension size  
			out_act (nn.Module, optional): Output activation function to replace regular activations
			out_params (dict): Parameters for the output activation function
			
		Returns:
			list: List of instantiated nn.Module layers ready to be used in nn.Sequential
		"""
		net = []  # List to store the instantiated layers
		flag_layer = False  # Flag to track if a layer with input/output dimensions has been added
		
		# Iterate through each layer class and its corresponding arguments
		for block, bargs in zip(blocks, block_args):
			# Create a copy to avoid modifying the original arguments dictionary
			bargs = bargs.copy()
			
			# Handle layers that require input/output dimensions (like Linear, Conv layers)
			if inp is not None and out is not None:
				# Check if this is a layer that takes input/output dimensions as first parameters
				if block.__name__ in self.layer_applied_names:
					# Instantiate the layer with input/output dimensions and additional arguments
					layer = block(inp, out, **bargs)
					net.append(layer)
					flag_layer = True  # Mark that we've added a dimensional layer
					continue

				# Handle activation functions - replace with out_act if provided
				if block.__name__ in self.activation_names:
					if out_act is not None:
						# If it's an activation function, use the provided out_act
						block = out_act
						bargs = out_params
					activation = block(**bargs)
					net.append(activation)
					self._init_weights(layer, act=activation)
					continue
				
				# Handle batch normalization layers that need num_features parameter
				if block.__name__ in nn.modules.batchnorm.__all__ or "InstanceNorm" in block.__name__:
					if flag_layer:
						# If we already added a dimensional layer, use output size for normalization
						bargs["num_features"] = out
					else:
						# If no dimensional layer yet, use input size for normalization
						bargs["num_features"] = inp
				
				# Handle layer normalization that needs normalized_shape parameter
				elif block.__name__ == "LayerNorm":
					if flag_layer:
						# If we already added a dimensional layer, use output size for normalization
						bargs["normalized_shape"] = out
					else:
						# If no dimensional layer yet, use input size for normalization
						bargs["normalized_shape"] = inp
			
			# Instantiate the layer with the (possibly modified) arguments
			net.append(block(**bargs))
		
		return net

	def _init_weights(self, layer, act):
		n_slope = 0
		act_name = act.__class__.__name__.lower()

		if act_name == "identity":
			act_name = "linear"
		elif act_name == "leakyrelu":
			act_name = "leaky_relu"
			n_slope = act.negative_slope
		elif act_name == "softmax":
			act_name = "linear"


		if self.weight_init == "uniform":
			inits = [nn.init.xavier_uniform_, nn.init.kaiming_uniform_]
			mode = "fan_in"
		elif self.weight_init == "normal":
			inits = [nn.init.xavier_normal_, nn.init.kaiming_normal_]
			mode = "fan_out"


		if act_name == "sigmoid" or act_name == "tanh":
			inits[0](layer.weight, gain = nn.init.calculate_gain(act_name))
		else:
			inits[1](layer.weight, mode = mode, nonlinearity = act_name, a = n_slope)


		if layer.bias is not None and self.weight_init is not None:
			nn.init.uniform_(layer.bias, a=-0.01, b=0.01)


