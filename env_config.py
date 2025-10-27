import torch
"""
DeepPy Configuration Management

This module provides a singleton configuration class to manage boolean flags
and settings across the entire DeepPy framework.
"""




class DeeppyEnvConfig:    
    def __init__(self):
        # Boolean flags for different features
        self.debug = False
        self.use_amp = False  # Automatic Mixed Precision
        self.torch_compile = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = "logs"
        self.checkpoint_dir = "checkpoints"
        
        self.gpu_prefetch = 1  # Number of prefetch streams
        self._xai = None  # Lazy initialization to avoid circular imports
        

    @property
    def xai(self):
        """Lazy initialization of XAI instance to avoid circular imports"""
        if self._xai is None:
            from .xai import Xai
            self._xai = Xai()
        return self._xai
        

    def print_config(self):
        """Print all current configuration flags"""
        flags = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        for flag, value in flags.items():
            print(f"{flag}: {value}")

# Global instance
env_config = DeeppyEnvConfig()