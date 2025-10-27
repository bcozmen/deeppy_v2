import torch
from torch.utils.data import Sampler



class UniquePerBatchSampler(Sampler):
    """
    A custom sampler that ensures batch-level uniqueness across multiple dataset repeats.
    
    This sampler generates multiple independent random permutations of dataset indices,
    but truncates each permutation to ensure the total number of samples is divisible
    by batch_size. This guarantees that:
    1. Each batch contains unique samples (no duplicates within a batch)
    2. No incomplete batches are created at the end
    3. Multiple passes through data maintain proper batch structure
    
    Args:
        dataset_size (int): Total number of samples in the dataset
        num_repeats (int): Number of times to repeat the dataset
        batch_size (int): Size of each batch - must divide evenly into samples
    
    Note:
        This sampler truncates samples to ensure len_per_epoch is divisible by batch_size.
        Some samples at the end of each epoch may be dropped to maintain batch integrity.
    """
    
    def __init__(self, dataset, batch_size, num_repeats):
        dataset_size = len(dataset)
        if batch_size > dataset_size:
            raise ValueError(
                f"Batch size ({batch_size}) cannot be greater than dataset size "
                f"({dataset_size}) for batch uniqueness guarantees."
            )
        
        self.dataset_size = dataset_size
        self.num_repeats = num_repeats
        self.batch_size = batch_size
        
        # Calculate samples per epoch that ensures complete batches only
        # This truncation is ESSENTIAL for batch uniqueness
        self.samples_per_epoch = (dataset_size // batch_size) * batch_size
    
    def __iter__(self):
        """
        Generate indices ensuring batch-level uniqueness.
        
        Creates num_repeats independent random permutations, each truncated to
        samples_per_epoch to ensure all batches are complete and contain unique samples.
        
        Returns:
            Iterator over indices with guaranteed batch uniqueness
        """
        # Generate multiple independent shuffled sequences
        # Each row is a complete shuffled pass through the dataset
        shuffled_indices = torch.stack([
            torch.randperm(self.dataset_size) 
            for _ in range(self.num_repeats)
        ])
        
        # CRITICAL: Truncate to ensure complete batches only
        # This maintains the invariant that each batch has unique samples
        truncated_indices = shuffled_indices[:, :self.samples_per_epoch]
        
        # Flatten to create a single sequence of indices
        all_indices = truncated_indices.flatten()
        
        # Return iterator (correct pattern for PyTorch samplers)
        return iter(all_indices.tolist())
    
    def __len__(self):
        """
        Return total number of samples across all repeats.
        
        Returns:
            int: samples_per_epoch * num_repeats (always divisible by batch_size)
        """
        return self.samples_per_epoch * self.num_repeats 
    