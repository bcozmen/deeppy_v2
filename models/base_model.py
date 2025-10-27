"""
DeepPy Base Model Framework

This module provides a comprehensive base class system for deep learning models with:
- Automatic device management and tensor conversion
- GPU memory prefetching and streaming
- Integrated logging and monitoring
- Abstract training/testing workflows
- State management and checkpointing
- Multi-network support with unified optimization

The framework uses metaclasses to automatically wrap methods for device management,
ensuring seamless GPU operations without explicit tensor transfers.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod, ABCMeta


from ..nn.network import Network
from ..env_config import env_config

# Logger class for managing training and evaluation logs.
class Logger():
	"""Logger class for managing training and evaluation logs.
	This class provides methods to initialize, add, and retrieve logs for different tags and keys.
	Args:
		tags (list): List of tags for different log categories (e.g., "Losses", "Metrics").
		keys (list): List of keys for each tag, defining the specific metrics to log.
	Example:
		logger = Logger(["Losses", "Metrics"], [["Total", "Recon"], ["Accuracy"]])
		logger.add("Losses", ["Total", "Recon"], [0.5, 0.2])
		logs = logger.get_metrics()
	"""
	def __init__(self, tags, keys):
		"""
		Initialize the Logger with specified tags and keys.
		"""
		self.tags = tags
		self.init_keys = keys
		self.reset()  # Initialize empty logs

	def reset(self):
		"""
		Reset the logs to empty state.
		"""
		self.logs = {}
		self.keys = {}
		for tag, key in zip(self.tags, self.init_keys):
			self.keys[tag] = key  # Store keys for each tag
			self.logs[tag] = {}
			for k in key:
				self.logs[tag][k] = []
		
	def add(self, tag, values):
		"""
		Add a new log entry for the specified tag and keys.
		Args:
			tag (str): The tag under which to log the values (e.g., "Losses", "Metrics").
			keys (list): List of keys for the values being logged.
			values (list): List of values corresponding to the keys.	
		"""
		for key,value in zip(self.keys[tag], values):
			self.logs[tag][key].append(value)
	def get(self):
		"""
		Retrieve the logged metrics, averaging over all recorded values.
		"""
		logs = {}
		for tag in self.logs.keys():
			for key in self.logs[tag].keys():
				logs.setdefault(tag, {})[key] = torch.stack(self.logs[tag][key]).mean(0)

		self.reset()  # Clear logs after retrieval
		return logs

class BaseClassMeta(type):
	"""
	Metaclass for automatic device management and method wrapping.
	
	This metaclass automatically wraps specified methods to ensure tensors are
	moved to the correct device before execution. It provides two types of wrapping:
	1. Direct device transfer (_to_device_methods)
	2. Stream-based prefetching (_to_device_methods_with_buffer)
	
	The metaclass also calls after_init() after object construction to finalize
	model setup (training mode, optimizer configuration, etc.).
	"""
	def __call__(cls, *args, **kwargs):
		# Create instance without running __init__
		obj = cls.__new__(cls, *args, **kwargs)

		# Call the actual __init__ from the child class
		cls.__init__(obj, *args, **kwargs)

		# Call base's after_init if it exists (for final setup)
		base_after_init = getattr(super(cls, obj), "after_init", None)
		if callable(base_after_init):
			base_after_init()

		# Wrap methods that need immediate device transfer
		transform_methods = getattr(obj, '_to_device_methods', [])
		for name in transform_methods:
			method = getattr(obj, name, None)
			if callable(method):
				def make_wrapper(func):
					def wrapped_func(self, X, *a, **kw):
						X = self.ensure(X)  # Move to device immediately
						r = func(X, *a, **kw)
						return r
					return wrapped_func
				wrapped = make_wrapper(method).__get__(obj)
				setattr(obj, name, wrapped)
		
		# Wrap methods that use GPU stream-based prefetching
		transform_methods = getattr(obj, '_to_device_methods_with_buffer', [])
		for name in transform_methods:
			method = getattr(obj, name, None)
			if callable(method):
				def make_wrapper(func):
					def wrapped_func(self, X, *a, **kw):
						X = self.ensure_with_stream(X)  # Queue for async transfer
						r = func(X, *a, **kw)
						return r
					return wrapped_func
				wrapped = make_wrapper(method).__get__(obj)
				setattr(obj, name, wrapped)
		return obj

class CombinedMeta(BaseClassMeta, ABCMeta):
	"""
	Combined metaclass that inherits from both BaseClassMeta and ABCMeta.
	
	This allows BaseModel to be both an abstract base class (with @abstractmethod)
	and have automatic device management capabilities.
	"""
	pass

class BaseModel(ABC, metaclass=CombinedMeta):
	"""
	Abstract base class for all deep learning models in the DeepPy framework.
	
	This class provides a comprehensive foundation for deep learning models with:
	
	**Key Features:**
	- **Automatic Device Management**: Tensors are automatically moved to GPU/CPU
	- **GPU Stream Processing**: Asynchronous data transfer and computation
	- **Multi-Network Support**: Handle multiple neural networks in one model
	- **Integrated Logging**: Built-in TensorBoard logging and XAI monitoring
	- **State Management**: Complete save/load functionality for checkpointing
	- **Abstract Training Loop**: Standardized optimize/test workflow
	
	**Method Wrapping:**
	The metaclass automatically wraps methods in _to_device_methods and
	_to_device_methods_with_buffer to handle tensor device transfers.
	
	**Abstract Methods (must be implemented by subclasses):**
	- get_loss(X): Compute loss for given input batch
	- back_propagate(loss): Handle backpropagation and optimizer steps
	- forward(X): Forward pass through the model
	- init_log_names(): Initialize logging metric names
	
	**Workflow:**
	1. Initialize model with networks and optimizers
	2. Call optimize(X) for training steps
	3. Call test(X) for evaluation steps
	4. Use save()/load() for checkpointing
	
	**Attributes:**
	- dependencies: List of required classes for this model
	- optimize_return_labels: Names of losses returned by optimization
	- _to_device_methods: Methods that need immediate tensor device transfer
	- _to_device_methods_with_buffer: Methods that use GPU streaming
	- device: PyTorch device (GPU/CPU)
	- criterion: Loss function(s)
	- training: Training/evaluation mode flag
	- nets: List of neural networks
	- optimizers: List of optimizers
	- xai: XAI monitoring instance
	- amp: Automatic Mixed Precision flag
	- gpu_prefetch: Number of batches to prefetch to GPU
	"""
	# Class-level configuration
	dependencies = []  # List of required classes for this model type
	optimize_return_labels = []  # Names of losses/metrics returned by optimize()

	# Methods that get automatic device transfer wrapping
	_to_device_methods = ["forward", "encode", "decode", "embed", "get_action"]
	_to_device_methods_with_buffer = ["optimize", "test"]  # Use GPU streaming
	

	def __init__(self, criterion=nn.MSELoss()):
		"""
		Initialize the base model with core components.
		
		Args:
			criterion: Loss function (default: MSELoss)
		"""
		# Core configuration from environment
		self.device = env_config.device
		self.xai = env_config.xai  # XAI monitoring system
		self.amp = env_config.use_amp  # Automatic Mixed Precision
		self.gpu_prefetch = env_config.gpu_prefetch  # Number of batches to prefetch
		
		# Model components
		self.criterion = criterion
		self.training = True
		
		# Logging and monitoring
		self.writer_losses, self.writer_metrics = [], []
		
		# Model architecture components
		self.nets = []  # List of neural networks
		self.params = []  # Initialization parameters for reconstruction
		self.objects = []  # Additional objects (criteria, etc.)
		self.optimizers = []  # List of optimizers
		
		# GPU streaming setup for efficient data transfer
		self.streams = [torch.cuda.Stream() for i in range(self.gpu_prefetch * 2 + 2)]
		self.train_queue, self.test_queue = [], []  # Queues for async data transfer
		
		
	def after_init(self):
		"""
		Post-initialization setup called by metaclass.
		
		This method is automatically called after __init__ to finalize model setup:
		- Set all networks to training mode
		- Configure optimizers
		- Initialize logging names
		- Set up parameter naming for monitoring
		"""
		self._init_networks()
		self._init_optimizers()
		self._init_loss_functions()
		self._init_logger()
		self._init_param_names()
		
		self.train()

		
		
		env_config.xai.register_model(self)
		
	def __call__(self, X):
		"""Make model callable - forwards to forward() method"""
		return self.forward(X)

	def __str__(self):
		"""String representation showing all networks"""
		return "\n=======================================\n".join([net.__str__() for net in self.nets])

	def optimize(self, X):
		"""
		Perform one training step with GPU streaming and prefetching.
		
		This method implements asynchronous training with GPU streams:
		1. Waits for sufficient batches in the training queue
		2. Processes batch with automatic mixed precision
		3. Computes loss and performs backpropagation
		4. Logs metrics if optimizer step occurred
		
		Args:
			X: Input batch (automatically queued by metaclass wrapper)
			
		Returns:
			bool: True if optimizer step was taken, False if still queuing
		"""
		if len(self.train_queue) < self.gpu_prefetch:
			return False  # Still queuing batches
			
		X, stream = self.train_queue.pop(0)
		with torch.cuda.stream(stream):
			with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
				loss = self.get_loss(X)
				loss_item = loss.item()
			optimizer_step = self.back_propagate(loss)
		self.streams.append(stream)  # Return stream to pool
		if optimizer_step:
			self.xai.notify(self)
		return optimizer_step
	
	@torch.no_grad()
	def test(self, X):
		"""
		Perform one evaluation step with GPU streaming.
		
		Similar to optimize() but without gradient computation:
		1. Waits for sufficient batches in test queue
		2. Processes batch in evaluation mode
		3. Computes loss for monitoring
		4. Logs evaluation metrics
		
		Args:
			X: Input batch (automatically queued by metaclass wrapper)
			
		Returns:
			bool: True if test step was completed, False if still queuing
		"""
		if len(self.test_queue) < self.gpu_prefetch:
			return False  # Still queuing batches
			
		X, stream = self.test_queue.pop(0)
		with torch.cuda.stream(stream):
			with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
				loss = self.get_loss(X)
		self.streams.append(stream)  # Return stream to pool
		
		self.xai.notify(self)
		return True

	#==========================================================================================
	# Abstract Methods - Must be implemented by subclasses
	@abstractmethod
	def get_loss(self, X):
		"""
		Compute loss for a given input batch.
		
		Args:
			X: Input batch (tuple of tensors typically)
			
		Returns:
			Loss tensor(s) - format depends on model implementation
		"""
		pass
		
	@abstractmethod
	def back_propagate(self, loss):
		"""
		Handle backpropagation and optimizer stepping.
		
		Args:
			loss: Loss tensor(s) from get_loss()
			
		Returns:
			bool: True if optimizer step was taken (for gradient accumulation)
		"""
		pass
		
	@abstractmethod
	def forward(self, X):
		"""
		Forward pass through the model.
		
		Args:
			X: Input tensor(s)
			
		Returns:
			Model output tensor(s)
		"""
		pass

	@abstractmethod
	def _init_networks(self):
		"""
		Initialize all networks in the model.
		
		This method should set self.nets to a list of Network instances.
		It is called automatically by the metaclass after __init__().
		"""
		pass
	@abstractmethod
	def _init_optimizers(self):
		"""
		Set up the optimizers list.
		
		This method should initialize self.optimizers with optimizer instances
		for each network in self.nets. It is called automatically by the metaclass
		after __init__() and after _init_networks().
		"""
		pass
	@abstractmethod
	def _init_loss_functions(self):
		"""
		Initialize loss functions from the objects list.
		
		This method should set self.objects to a list of loss function instances
		(e.g., nn.MSELoss, nn.CrossEntropyLoss). It is called automatically by the
		metaclass after __init__() and after _init_optimizers().
		"""
		pass

	@abstractmethod
	def _init_logger(self):
		"""
		Initialize the logger for training and evaluation metrics.
		This method should set up the Logger instance with appropriate tags and keys
		use self.create_logger() to create a Logger instance.
		"""
		pass

	@staticmethod
	def create_logger(tags=None, keys=None):
		"""
		Factory method to create a Logger instance.
		"""
		return Logger(tags, keys)

	@abstractmethod
	def _load_loss_functions(self):
		"""Example self.criterion1 , self.criterion2 = self.loss_functions"""
		pass
	@abstractmethod
	def _init_optimizers(self):
		"""
		self.optimizers = [optimizer1, optimizer2, ...]
		"""
	#==========================================================================================
	# Training/Evaluation Mode Control

	def train(self):
		"""Set all networks to training mode and update training flag"""
		[net.train() for net in self.nets]
		self.training = True	
	def eval(self):
		"""Set all networks to evaluation mode and update training flag"""
		[net.eval() for net in self.nets]
		self.training = False

	#==========================================================================================
	# Tensor Device Management
	
	def ensure_tensor_device(self, X):
		"""
		Ensure data is a PyTorch tensor on the correct device.
		
		Args:
			X: Input data (can be tensor, numpy array, list, etc.)
			
		Returns:
			Tensor on the correct device, or None if input is None
		"""
		if X is None:
			return X
		if not torch.is_tensor(X):
			X = torch.tensor(X)
		if X.device != self.device:
			X = X.to(self.device, non_blocking=True)
		return X
	def ensure(self, X):
		"""
		Handle device transfer for single tensors or tuples of tensors.
		
		Args:
			X: Single tensor or tuple of tensors
			
		Returns:
			Tensor(s) on correct device
		"""
		if isinstance(X, tuple):
			X = tuple(map(self.ensure_tensor_device, X))
		else:
			X = self.ensure_tensor_device(X)
		return X
	def ensure_with_stream(self, X):

		"""
		Handle device transfer using GPU streams for asynchronous processing.
		
		This method:
		1. Gets a stream from the pool
		2. Transfers data to device within the stream context
		3. Adds (data, stream) to appropriate queue for later processing
		
		Args:
			X: Input data to transfer
		"""
		stream = self.streams.pop(0)
		with torch.cuda.stream(stream):
			if isinstance(X, tuple):
				X = tuple(map(self.ensure_tensor_device, X))
			else:
				X = self.ensure_tensor_device(X)
		
		# Add to appropriate queue based on current mode
		queue = self.train_queue if self.training else self.test_queue
		queue.append((X, stream))
	#==========================================================================================
	# Model Initialization and Configuration
	def _init_param_names(self):
		"""
		Initialize descriptive names for all model parameters for monitoring.
		
		Creates a hierarchical naming scheme for parameters:
		- Net{index}/layer_group/layer_name/parameter_name
		- Handles both regular and compiled models (_orig_mod prefix)
		- Sets 'dpname' attribute on each parameter for logging
		"""
		for net_ix, net in enumerate(self.nets):
			for n, p in net.named_parameters():
				net_pre = f"Net{net_ix}/"
				flag = "layers" in n
				n = n.replace("_orig_mod.", "").split(".")
				
				b = "".join(n[:2]) + "/"
				n = n[2:]
				if flag:
					b += "".join(n[:2]) + "/"
					n = n[2:]
				
				p.dpname = net_pre + b + ".".join(n)

	@property
	def global_step(self):
		return self.optimizers[0]._optimizer_steps_counter - 1
	def scheduler_step(self):
		"""Take a scheduler step for all networks"""
		for optimizer in self.optimizers:
			optimizer.scheduler_step()

	#===========================================================================================
	# Saving and loading states
	def save(self,file_name = None, return_dict=False):
		#Save the model given a file name

		
		optimizer_dicts = [optimizer.save_states() for optimizer in self.optimizers]
		network_dicts = [net.save_states() for net in self.nets]
		save_dict = {
			"params" : self.params,
			"nets" : network_dicts,
			"loss_functions" : self.loss_functions,
			"optimizer" : optimizer_dicts
		}
		if return_dict:
			return save_dict
		torch.save(save_dict, file_name + "/model.pt")
	
	@classmethod
	def load(clss, file_name):
		#Load the model from the class.
		#First initialize a new object, and then load the checkpoint
		if isinstance(file_name, dict):
			checkpoint = file_name
		else:
			checkpoint = torch.load(file_name + "/model.pt", weights_only = False)
		params = checkpoint["params"]
		network_dicts = checkpoint["nets"]
		loss_functions = checkpoint["loss_functions"]
		optimizer_dicts = checkpoint["optimizer"]



		instance = clss(*params)

		for net,net_dicts in zip(instance.nets, network_dicts):
			net.load_states(net_dicts)

		if optimizer_dicts is not None:
			[optimizer.load_states(dict) for optimizer, dict in zip(clss.optimizers, optimizer_dicts)]
		
		instance.loss_functions = loss_functions
		instance._load_loss_functions()
		return instance
	
	def save_states(self):
		#Helper function to save
		return self.save(return_dict = True)

	def load_states(self, dic):
		#Helper function to load
		params = dic["params"]
		dicts = dic["nets"]
		objs = dic["objs"]
		for net,net_dicts in zip(self.nets, dicts):
			net.load_states(net_dicts)

	#===========================================================================================



	def param_norm(self):
		total_norms = []
		for model in self.nets:
			for p in model.parameters():
				if p is not None and p.requires_grad:
					total_norms.append(p.data.norm(2).item())

		return total_norms
	def last_lr(self):
		#Net the last_lr if a scheduler is used
		try:
			return [optimizer.scheduler.scheduler.get_last_lr()[0] for optimizer in self.optimizers]
		except:
			return False
	def print_param_count(self):
		[net.print_param_count() for net in self.nets]

	def print_param_size(self):
		[net.print_param_size() for net in self.nets]

	def print_param_info(self):
		for ix,net in enumerate(self.nets):
			param_count, param_size = net.get_param_info()
			print(f"Net : {ix}")
			print(f"    Parameters : {param_count/(1e6):6.4f} M")
			print(f"    Size       : {param_size / (1024**3):6.4f} GB")
			
	
	
	



class Model(BaseModel):
	"""
	Concrete implementation of BaseModel for simple single-network models.
	
	This class provides a basic implementation that:
	- Uses a single Network instance
	- Implements standard forward pass
	- Computes MSE loss between predictions and targets
	- Delegates backpropagation to the network's optimizer
	
	This serves as both a working model and an example of how to implement
	the abstract methods from BaseModel.
	
	Args:
		network_params (dict): Parameters to initialize the Network
		device: PyTorch device (GPU/CPU)
		criterion: Loss function (default: MSELoss)
		amp (bool): Enable Automatic Mixed Precision
	"""
	dependencies = [Network]
	optimize_return_labels = ["Loss"]
	
	def __init__(self, network_params, device=None, criterion=nn.MSELoss(), amp=True):
		super().__init__(device=device, criterion=criterion, amp=amp)

		# Initialize single network and move to device
		self.net = Network(**network_params).to(self.device)
		
		# Store initialization parameters for save/load
		self.params = [network_params, device]
		self.nets = [self.net]
		self.objects = [criterion]

	def forward(self, X):
		"""Forward pass through the single network"""
		return self.net(X)
	
	def get_loss(self, X):
		"""
		Compute loss for input batch.
		
		Args:
			X: Tuple of (input, target) tensors
			
		Returns:
			Tuple of (loss_tensor, loss_value)
		"""
		X, y = X
		outs = self(X)
		loss = self.criterion(outs, y)
		return loss, loss.item()

	def back_propagate(self, loss):
		"""Delegate backpropagation to the network's optimizer"""
		self.net.back_propagate(loss)

	