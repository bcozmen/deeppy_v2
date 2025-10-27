"""
DeepPy Optimizer Module

This module provides wrapper classes for PyTorch optimizers, schedulers, and gradient clippers
to create a unified training interface with built-in support for:
- Automatic Mixed Precision (AMP)
- Gradient clipping
- Learning rate scheduling
- Gradient accumulation
- Logging and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import GradScaler
from ..env_config import env_config


class Scheduler():
	"""
	A scheduler wrapper that provides automatic stepping functionality.
	
	This wrapper allows schedulers to automatically step after each optimizer step
	or be manually controlled based on the auto_step parameter.
	
	Args:
		optimizer: PyTorch optimizer instance
		scheduler: PyTorch scheduler class (e.g., torch.optim.lr_scheduler.StepLR)
		auto_step (bool): If True, scheduler steps automatically with optimizer
		**kwargs: Additional arguments passed to the scheduler constructor
	"""
	def __init__(self, optimizer, scheduler, auto_step=True, **kwargs):
		self.auto_step = auto_step
		self.scheduler = scheduler(optimizer, **kwargs)
		
	def step(self):
		"""Execute one scheduler step"""
		self.scheduler.step()

class Clipper():
	"""
	A gradient clipping wrapper that provides a unified interface for different clipping methods.
	
	This wrapper encapsulates gradient clipping functionality and stores clipping parameters
	for consistent application across training steps.
	
	Args:
		clipper: Gradient clipping function (e.g., torch.nn.utils.clip_grad_norm_)
		**kwargs: Parameters for the clipping function (e.g., max_norm, norm_type)
	
	Example:
		clipper = Clipper(torch.nn.utils.clip_grad_norm_, max_norm=1.0)
		clipper(model.parameters())
	"""
	def __init__(self, clipper, **kwargs):
		self.clipper = clipper
		self.clipper_params = kwargs

	def __call__(self, parameters):
		"""Apply gradient clipping to the given parameters"""
		
		if isinstance(parameters, list):
			param = []
			for group in parameters:
				param.extend(group["params"])
			parameters = param
		self.clipper(parameters, **self.clipper_params)

#TODO: Verify optimizer save/load functionality is correct
class Optimizer():
	"""
	A comprehensive optimizer wrapper that integrates multiple training components.
	
	This class combines PyTorch optimizers with:
	- Gradient accumulation for effective large batch training
	- Automatic Mixed Precision (AMP) support via GradScaler
	- Learning rate scheduling with automatic or manual stepping
	- Gradient clipping for training stability
	- Integrated logging and monitoring via XAI module
	
	The optimizer handles the complete training step workflow:
	1. Loss scaling and backward pass
	2. Gradient accumulation tracking
	3. Gradient unscaling (for AMP)
	4. Logging of optimizer statistics
	5. Gradient clipping (if enabled)
	6. Optimizer and scheduler stepping
	
	Args:
		model_parameters: Model parameters to optimize (from model.parameters())
		optimizer: PyTorch optimizer class (default: AdamW)
		optimizer_args (dict): Arguments passed to optimizer constructor
		gradient_accumulation_steps (int): Number of steps to accumulate gradients
		clipper (Clipper): Gradient clipping wrapper instance
		scheduler_params (dict): Parameters for learning rate scheduler
	
	Example:
		optimizer = Optimizer(
			model.parameters(),
			optimizer=torch.optim.AdamW,
			optimizer_args={'lr': 1e-4, 'weight_decay': 0.01},
			gradient_accumulation_steps=4,
			clipper=Clipper(torch.nn.utils.clip_grad_norm_, max_norm=1.0),
			scheduler_params={'scheduler': torch.optim.lr_scheduler.StepLR, 'step_size': 10}
		)
	"""
	def __init__(self, model_parameters, optimizer=optim.AdamW, optimizer_args={},  gradient_accumulation_steps=1,
			  clipper=None, clipper_params = None, scheduler_params=None):
		
		# Store configuration parameters
		self.model_parameters = model_parameters
		self.optimizer_args = optimizer_args
		self.scheduler_params = scheduler_params
		self.gradient_accumulation_steps = int(gradient_accumulation_steps)
		
		# Initialize step counters
		self._step_counter = 0  # Tracks backward passes for gradient accumulation
		self._optimizer_steps_counter = 0  # Tracks actual optimizer steps
		
		# Initialize variables from environment config
		self.xai = None
		self.scaler = GradScaler(enabled=env_config.use_amp)
		
		
		
		# Initialize the underlying PyTorch optimizer
		self.optimizer = optimizer(model_parameters, **optimizer_args)
		self.optimizer.zero_grad(set_to_none=True)

		# Store gradient clipper (can be None if no clipping desired)
		self.clipper = clipper
		if clipper is not None:
			if clipper_params is None:
				raise ValueError("clipper_params must be provided if clipper is not None")
			# Initialize the clipper with parameters if provided
			self.clipper = Clipper(clipper, **clipper_params)

		# Initialize learning rate scheduler if parameters provided
		self.scheduler = None
		if scheduler_params is not None:
			self.scheduler = Scheduler(self.optimizer, **scheduler_params) 
	
	def register_xai(self, xai):
		"""
		Register an XAI instance for logging and monitoring.
		
		This allows the optimizer to log statistics and metrics during training.
		
		Args:
			xai_instance: An instance of the Xai class for logging
		"""
		self.xai = xai
	def step(self, loss):
		"""
		Perform one optimization step with gradient accumulation support.
		
		This method handles the complete optimization workflow:
		1. Scales loss for gradient accumulation
		2. Performs backward pass with AMP scaling
		3. Accumulates gradients over multiple steps
		4. Unscales gradients and applies clipping
		5. Steps optimizer and scheduler
		6. Logs optimization statistics
		
		Args:
			loss: Scalar loss tensor to backpropagate
			
		Returns:
			bool: True if optimizer step was taken, False if still accumulating
		"""
		# Scale loss by accumulation steps to maintain effective batch size
		loss = loss / self.gradient_accumulation_steps
		# Perform backward pass with gradient scaling for AMP
		self.scaler.scale(loss).backward()
		self._step_counter += 1
		
		# Check if we've accumulated enough gradients
		if (self._step_counter) % self.gradient_accumulation_steps != 0:
			return False  # Continue accumulating, don't step optimizer yet
			
		# Unscale gradients before clipping (required for AMP)
		self.scaler.unscale_(self.optimizer) 
		
		# Log optimizer statistics before applying updates
		self.xai.notify(self)

		# Apply gradient clipping if configured
		if self.clipper is not None:
			self.clipper(self.model_parameters)
			
		# Perform optimizer step with gradient scaling
		self.scaler.step(self.optimizer)
		self.scaler.update()  # Update the scale factor for next iteration
		self.optimizer.zero_grad(set_to_none=True)  # Clear gradients efficiently

		# Step the learning rate scheduler if auto-stepping is enabled
		if self.scheduler is not None and self.scheduler.auto_step:
			self.scheduler.step()
		
		self._optimizer_steps_counter += 1
		return True  # Optimizer step was taken


	def save_states(self):
		"""
		Save all optimizer-related states for checkpointing.
		
		Returns:
			dict: Dictionary containing all necessary state information:
				- optimizer: Optimizer state dict
				- clipper: Gradient clipper instance
				- scheduler: Scheduler state dict (if scheduler exists)
				- _optimizer_steps_counter: Number of optimizer steps taken
				- _step_counter: Number of backward passes performed
		"""
		# Handle scheduler state - can be None if no scheduler is configured			
		return {
			"optimizer": self.optimizer.state_dict(),
			"clipper": self.clipper,
			"scheduler": self.scheduler,
			"_optimizer_steps_counter": self._optimizer_steps_counter,
			"_step_counter": self._step_counter,
		}

	def load_states(self, state_dict):
		"""
		Load optimizer states from a checkpoint.
		
		Args:
			state_dict (dict): State dictionary from save_states()
		"""
		# Restore gradient clipper
		self.clipper = state_dict["clipper"]
		
		# Restore optimizer state and clear gradients
		self.optimizer.load_state_dict(state_dict["optimizer"])
		self.optimizer.zero_grad(set_to_none=True)
		
		# Restore step counters
		self._optimizer_steps_counter = state_dict["_optimizer_steps_counter"]
		self._step_counter = state_dict["_step_counter"]
		
		self.scheduler = state_dict["scheduler"]
		
	def scheduler_step(self):
		self.scheduler.step() if self.scheduler is not None else None

