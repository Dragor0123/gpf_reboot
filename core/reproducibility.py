"""Reproducibility utilities for consistent experiment results."""

import random
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_reproducible_seeds(seed: int = 42, strict_deterministic: bool = True) -> None:
    """Set all random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
        strict_deterministic: Whether to enforce strict deterministic behavior
                             (may impact performance)
    """
    logger.info(f"Setting reproducible seeds with seed={seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if strict_deterministic:
            # Ensure deterministic behavior (may be slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Enabled CUDA deterministic mode (may impact performance)")
        else:
            # Allow non-deterministic behavior for better performance
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled CUDA benchmark mode for better performance")
    
    # Set environment variables for additional determinism
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info("✅ All random seeds set for reproducibility")


def get_random_state() -> dict:
    """Get current random state for all generators.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """Restore random state for all generators.
    
    Args:
        state: Dictionary containing random states (from get_random_state)
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])
    torch.set_rng_state(state['torch_random'])
    
    if torch.cuda.is_available() and 'torch_cuda_random' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda_random'])
    
    logger.info("Restored random state")


class ReproducibilityContext:
    """Context manager for temporary reproducibility settings."""
    
    def __init__(self, seed: Optional[int] = None, strict_deterministic: bool = True):
        self.seed = seed
        self.strict_deterministic = strict_deterministic
        self.original_state = None
        self.original_cuda_settings = {}
    
    def __enter__(self):
        # Save current state
        self.original_state = get_random_state()
        
        # Save CUDA settings
        if torch.cuda.is_available():
            self.original_cuda_settings = {
                'deterministic': torch.backends.cudnn.deterministic,
                'benchmark': torch.backends.cudnn.benchmark,
            }
        
        # Set new seed if provided
        if self.seed is not None:
            set_reproducible_seeds(self.seed, self.strict_deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.original_state:
            set_random_state(self.original_state)
        
        # Restore CUDA settings
        if torch.cuda.is_available() and self.original_cuda_settings:
            torch.backends.cudnn.deterministic = self.original_cuda_settings['deterministic']
            torch.backends.cudnn.benchmark = self.original_cuda_settings['benchmark']


def create_reproducible_dataloader(dataset, batch_size: int, shuffle: bool = True, 
                                 seed: Optional[int] = None, **kwargs):
    """Create a DataLoader with reproducible shuffling.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seed: Seed for reproducible shuffling
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader with reproducible behavior
    """
    from torch.utils.data import DataLoader
    
    if shuffle and seed is not None:
        # Create generator with fixed seed for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs['generator'] = generator
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def verify_reproducibility(func, *args, num_runs: int = 3, **kwargs):
    """Verify that a function produces reproducible results.
    
    Args:
        func: Function to test
        *args: Function arguments
        num_runs: Number of runs to compare
        **kwargs: Function keyword arguments
        
    Returns:
        bool: True if all runs produce identical results
        
    Raises:
        AssertionError: If results are not reproducible
    """
    logger.info(f"Verifying reproducibility over {num_runs} runs")
    
    results = []
    original_state = get_random_state()
    
    for i in range(num_runs):
        # Reset to original state before each run
        set_random_state(original_state)
        
        # Run function
        result = func(*args, **kwargs)
        results.append(result)
        
        logger.info(f"Run {i+1} completed")
    
    # Compare results
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        if isinstance(first_result, torch.Tensor):
            if not torch.allclose(first_result, result, rtol=1e-6, atol=1e-8):
                raise AssertionError(f"Results differ between run 1 and run {i+1}")
        elif isinstance(first_result, (list, tuple)):
            for j, (a, b) in enumerate(zip(first_result, result)):
                if isinstance(a, torch.Tensor):
                    if not torch.allclose(a, b, rtol=1e-6, atol=1e-8):
                        raise AssertionError(f"Results differ at position {j} between run 1 and run {i+1}")
                elif a != b:
                    raise AssertionError(f"Results differ at position {j} between run 1 and run {i+1}")
        elif first_result != result:
            raise AssertionError(f"Results differ between run 1 and run {i+1}")
    
    logger.info("✅ Reproducibility verified successfully")
    
    # Restore original state
    set_random_state(original_state)
    
    return True