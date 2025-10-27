import torch
from ..env_config import env_config
 
class Xai:
	def __init__(self):
		self.registered_models = []
		self.writer = None
	
	def _init_writer(self):
		"""
		Initialize the writer for logging.
		
		This method sets up the writer instance for logging statistics and metrics.
		It can be extended to support different logging backends (e.g., TensorBoard, WandB).
		"""
		from torch.utils.tensorboard import SummaryWriter
		self.writer = SummaryWriter(log_dir=env_config.log_dir)
		if env_config.debug:
			print("[Xai] Writer initialized")
		
	def register_model(self, model):
		"""Register a model instance for XAI purposes."""
		if model in self.registered_models:
			if env_config.debug:
				print(f"[Xai] Model already registered: {model.__class__.__name__}")
			return
		
		[optimizer.register_xai(self) for optimizer in model.optimizers]
		optimizers = model.optimizers
		nets = model.nets
		self.registered_models.append((model,optimizers,nets))

		if env_config.debug:
			print(f"[Xai] Model registered: {model.__class__.__name__}")
		return
	
	def notify(self, object):
		"""
		Notify the XAI system about an event or object.
		
		This can be used to log statistics, metrics, or any other relevant information.
		
		Args:
			object: The object to notify (e.g., optimizer, model)
		"""
		if self.writer is None:
			self._init_writer()

		# Import here to avoid circular imports
		from ..nn import Optimizer
		from ..models import BaseModel

		if isinstance(object, Optimizer):
			self.log_optimizer(object)
		elif isinstance(object, BaseModel):
			self.log_model(object)

	def log_optimizer(self, optimizer):
		"""
		Log optimizer statistics and metrics.
		
		This method collects and logs various statistics from the optimizer,
		such as parameter norms, gradient norms, and learning rates.
		
		Args:
			optimizer: The optimizer instance to log
		"""
		# Calculate gradient norm
		metrics = {}
		

		for group in optimizer.optimizer.param_groups:
			for p in group["params"]:
				if p.grad is not None:
					param_norm, grad_norm = p.data.detach().norm(2).item(), p.grad.detach().norm(2).item()
					metrics.setdefault("param_norms", []).append(param_norm)
					metrics.setdefault("grad_norms", []).append(grad_norm)

				
				state = optimizer.optimizer.state[p]

				for key in state.keys():
					if "step" in key:
						continue
					metrics.setdefault(key, []).append(state[key].detach().cpu().norm(2).item())

		for key in metrics.keys():
			if isinstance(metrics[key], list):
				data = torch.tensor(metrics[key])
			if len(data) > 0:
				self.writer.add_histogram(f"Optimizer/{key}", data, optimizer._optimizer_steps_counter)

	def log_model(self, model):
		"""
		Log model statistics and metrics.
		
		This method collects and logs various statistics from the model,
		such as parameter norms, gradient norms, and learning rates.
		
		Args:
			model: The model instance to log
		"""
		logs = model.logger.get()

		mode = "Train" if model.training else "Test"


		for tag in logs.keys():
			self.writer.add_scalars(f"{mode}/{tag}", logs[tag], model.global_step)
			
		if model.training:
			self.log_detail(model)

		if env_config.debug:
			print(f"[Xai] Logging model: {model.__class__.__name__}")
	
	def log_detail(self, model):
		"""
		Log detailed statistics for the model.
		
		This method can be extended to log additional details about the model,
		such as layer-wise statistics, activation distributions, etc.
		
		Args:
			model: The model instance to log
		"""
		if env_config.debug:
			print(f"[Xai] Logging detailed information for model: {model.__class__.__name__}")
		
		# Placeholder for detailed logging logic
		pass

		