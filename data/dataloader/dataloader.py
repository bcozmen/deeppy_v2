from .dataloader_base import DataLoaderBase

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import random
from collections import deque, namedtuple
import pickle



class DatasetLoader(DataLoaderBase):
    """
    Concrete implementation of DatasetLoader for automatic train/test/validation splitting.
    
    This class takes a dataset and automatically splits it into train/test/validation sets
    with configurable ratios. Supports saving/loading split indices for reproducibility.
    
    Features:
    - Automatic data splitting with configurable ratios
    - Custom sampler support (e.g., for data augmentation)
    - Save/load functionality for reproducible splits
    - Handles edge cases (empty splits, small datasets)
    """
    def __init__(self, data, splits=[0.8, 0.1, 0.1], file_name=None, 
                 sampler=None, sampler_args=None,
                 batch_size=64, dataloader_args=None):
        """
        Initialize dataset with automatic splitting.
        
        Args:
            data: PyTorch dataset to be split
            splits (list): Ratios for [train, test, valid] splits (must sum to ~1.0)
            file_name (str): If provided, load splits from this path instead of creating new ones
            sampler (class): Custom sampler class for data loading
            sampler_args (dict): Arguments to pass to the sampler
            batch_size (int): Batch size for all data loaders
            dataloader_args (dict): Additional arguments for PyTorch DataLoader
        """
        super().__init__(batch_size=batch_size, dataloader_args=dataloader_args, sampler=sampler)        

        self.data = data
        self.sampler_args = sampler_args or {}
        
        # If file_name provided, load existing splits instead of creating new ones
        if file_name is not None:
            self.load(data, file_name)
            return
        
        # Validate and normalize splits
        if len(splits) != 3:
            raise ValueError("Splits must be a list of three values for train, test, and valid sets.")
        if not torch.is_tensor(splits):
            splits = torch.tensor(splits, dtype=torch.float32)        
        self.splits = splits / splits.sum()  # Normalize to ensure splits sum to 1.0
        
        self._prepare()
        
    def _prepare(self):
        """Set up datasets and data loaders for all splits."""
        self._prepare_splits(self.data)
        self.train_loader = self._prepare_dataloader(self.train_dataset)
        self.test_loader = self._prepare_dataloader(self.test_dataset)
        self.valid_loader = self._prepare_dataloader(self.valid_dataset)

    def _prepare_splits(self, data):
        """
        Split the dataset into train/test/validation sets.
        
        Uses random_split with calculated lengths. Handles rounding by giving
        any remainder samples to the training set.
        """
        total_length = len(data)
        lengths = torch.floor(self.splits * total_length).to(torch.int64)
        
        # Add any remainder to training set to ensure all samples are used
        lengths[0] += total_length - lengths.sum()

        self.train_dataset, self.test_dataset, self.valid_dataset = random_split(
            data, lengths.tolist()
        )
    
    def _prepare_dataloader(self, dataset):
        """
        Create a DataLoader for the given dataset split.
        
        Args:
            dataset: PyTorch dataset (could be empty)
            
        Returns:
            DataLoader or None if dataset is empty
        """
        if dataset is None or len(dataset) == 0:
            return None
        
        # Ensure batch size doesn't exceed dataset size
        effective_batch_size = min(len(dataset), self.batch_size)
        
        # Create dataloader arguments copy to avoid modifying original
        loader_args = self.dataloader_args.copy()
        
        # Add custom sampler if specified
        if self.sampler is not None: 
            loader_args['sampler'] = self.sampler(
                dataset, 
                batch_size=effective_batch_size, 
                **self.sampler_args
            )
            
        return DataLoader(dataset, batch_size=effective_batch_size, **loader_args)

    def save(self, file_name):
        """
        Save dataset split indices for reproducibility.
        
        Args:
            file_name (str): Path to save split indices (will create split_indices.pkl)
        """
        split_indices = {
            "train": self.train_dataset.indices,
            "test": self.test_dataset.indices,
            "valid": self.valid_dataset.indices
        }
        torch.save(split_indices, f"{file_name}/split_indices.pkl")

    def load(self, data, file_name):
        """
        Load dataset splits from saved indices.
        
        Args:
            data: Original dataset to apply splits to
            file_name (str): Path containing split_indices.pkl
        """
        split_indices = torch.load(f"{file_name}/split_indices.pkl", weights_only=False)
        
        # Recreate dataset splits using saved indices
        self.train_dataset = Subset(data, split_indices["train"])
        self.test_dataset = Subset(data, split_indices["test"])
        self.valid_dataset = Subset(data, split_indices["valid"])

        # Create data loaders for all splits
        self.train_loader = self._prepare_dataloader(self.train_dataset)
        self.test_loader = self._prepare_dataloader(self.test_dataset)
        self.valid_loader = self._prepare_dataloader(self.valid_dataset)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))

class ReplayBuffer():
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def __getitem__(self, index):   
        return self.buffer[index]
    def push(self, args):
        self.buffer.extend([Transition(*k) for k in zip(*args)])

    def sample(self, batch_size):
        return tuple([torch.stack(k) for k in (zip(*random.sample(self.buffer,batch_size)))])

class EnvDatasetLoader(DataLoaderBase):
    def __init__(self, env, buffer_size = 20000, batch_size = 128, start_size = 128):
        super().__init__( batch_size = batch_size)
        self.memory = ReplayBuffer(buffer_size)
        self.env = env

        try:
            self.env.action_space.n
            self.action_item = True
        except:
            self.action_item = False

        self.start_size = start_size
        self.reset()

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.state = torch.tensor(self.env.reset()[0], dtype=torch.float32).unsqueeze(0)
    
    def train_data(self):
        if len(self.memory) < self.start_size:
            return None
        return  self.memory.sample(self.batch_size)

    def collect(self, model):
        action = model(self.state).to(self.device)

        env_action = action.squeeze(0).numpy()
        if self.action_item:
            env_action = env_action.item()
        observation, reward, termination, truncation, data = self.env.step(env_action)

        done = torch.tensor([(termination or truncation)], dtype=torch.bool).unsqueeze(0)
        reward = torch.tensor([reward],dtype = torch.float32).unsqueeze(0)
        next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        # Store the transition in memory
        if model.training:    
            self.memory.push((self.state, action, next_state, reward,done))

        # Move to the next state
        self.state = next_state
        if done.item():
            self.reset()
        return done.item(), reward.item()

    def save(self,file_name):
        with open(file_name + '/memory.pkl', 'wb') as f:
            pickle.dump(self.memory.buffer, f)

    def load(self, file_name):
        with open(file_name + '/memory.pkl', 'rb') as f:
            self.memory.buffer = pickle.load(f)

    def emulate(self,model):
        counter = 0
        cum_reward = 0
        done = False

        self.reset()
        model.eval()
        while(not done):
            done ,reward = self.collect(model)
            counter += 1
            cum_reward += reward
        self.reset()
        return counter, cum_reward
