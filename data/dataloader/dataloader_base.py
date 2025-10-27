from abc import ABC, abstractmethod
import torch


class DataLoaderBase(ABC):
    """
    Abstract base class for dataset management with train/test/validation splits.
    
    Provides a unified interface for dataset handling with automatic iterator management
    and convenient methods for accessing batches from different data splits.
    
    Features:
    - Automatic iterator reset when exhausted
    - Generic batch retrieval for all splits
    - Support for custom samplers
    - Configurable batch sizes and dataloader arguments
    """
    
    
    def __init__(self, batch_size=64, dataloader_args=None, sampler=None):
        """
        Initialize the dataset base.
        
        Args:
            batch_size (int): Default batch size for all data loaders
            dataloader_args (dict): Additional arguments for PyTorch DataLoader
            sampler (class): Custom sampler class (not instance)
        """
        self.dataloader_args = dataloader_args or {}
        self.batch_size = batch_size
        self.sampler = sampler

        # Initialize datasets and loaders as None
        self.train_dataset, self.test_dataset, self.valid_dataset = None, None, None
        self.train_loader, self.test_loader, self.valid_loader = None, None, None
        
        # Dictionary to manage iterators elegantly - avoids StopIteration handling repetition
        self._iterators = {}

    def __len__(self):
        """Return total number of samples across all splits."""
        datasets = [self.train_dataset, self.test_dataset, self.valid_dataset]
        return sum(len(dataset) for dataset in datasets if dataset is not None)

    def _get_next_batch(self, loader_name):
        """
        Generic method to get next batch from any loader with automatic reset.
        
        This method handles the iterator lifecycle automatically:
        1. Creates iterator on first access
        2. Returns next batch from iterator
        3. Resets iterator when exhausted and returns first batch of new cycle
        
        Args:
            loader_name (str): Name of the loader ('train', 'test', 'valid')
            
        Returns:
            Batch data from the specified loader
            
        Raises:
            ValueError: If the specified loader is not available
        """
        loader = getattr(self, f"{loader_name}_loader")
        if loader is None:
            raise ValueError(f"No {loader_name} loader available")
            
        # Get or create iterator for this loader
        if loader_name not in self._iterators:
            self._iterators[loader_name] = iter(loader)
        
        try:
            return next(self._iterators[loader_name])
        except StopIteration:
            # Reset iterator and get first batch of new epoch
            self._iterators[loader_name] = iter(loader)
            return next(self._iterators[loader_name])

    def train_data(self):
        """Get next training batch with automatic epoch cycling."""
        return tuple(self._get_next_batch('train'))

    def test_data(self):
        """Get next test batch with automatic epoch cycling."""
        return tuple(self._get_next_batch('test'))

    def valid_data(self):
        """Get next validation batch with automatic epoch cycling."""
        return tuple(self._get_next_batch('valid'))

    @abstractmethod
    def save(self, file_name):
        """Save dataset splits to file. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def load(self, file_name):
        """Load dataset splits from file. Must be implemented by subclasses."""
        pass