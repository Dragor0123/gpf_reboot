"""Device management utilities."""

import torch
import logging
from typing import Union

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and optimization."""
    
    def __init__(self):
        self._device_cache = {}
    
    def get_device(self, device_str: str = "auto") -> torch.device:
        """Get appropriate device for computation.
        
        Args:
            device_str: Device specification ("auto", "cuda", "cpu", or specific device)
            
        Returns:
            PyTorch device object
        """
        if device_str in self._device_cache:
            return self._device_cache[device_str]
        
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif device_str == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU device")
        else:
            # Specific device string (e.g., "cuda:1")
            try:
                device = torch.device(device_str)
                logger.info(f"Using specified device: {device}")
            except RuntimeError as e:
                logger.error(f"Invalid device specification: {device_str}, error: {e}")
                device = torch.device("cpu")
                logger.info("Falling back to CPU")
        
        # Cache the result
        self._device_cache[device_str] = device
        return device
    
    def get_device_info(self, device: Union[torch.device, str, None] = None) -> dict:
        """Get detailed information about a device.
        
        Args:
            device: Device to inspect (if None, uses current default)
            
        Returns:
            Dictionary with device information
        """
        if device is None:
            device = self.get_device("auto")
        elif isinstance(device, str):
            device = self.get_device(device)
        
        info = {
            "device_type": device.type,
            "device_index": device.index,
            "device_str": str(device)
        }
        
        if device.type == "cuda" and torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_name": torch.cuda.get_device_name(device),
                "memory_allocated": torch.cuda.memory_allocated(device),
                "memory_reserved": torch.cuda.memory_reserved(device),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device),
                "device_capability": torch.cuda.get_device_capability(device),
            })
        
        return info
    
    def optimize_for_device(self, device: torch.device, deterministic: bool = True) -> None:
        """Optimize PyTorch settings for the given device.
        
        Args:
            device: Target device
            deterministic: Whether to prioritize deterministic behavior
        """
        if device.type == "cuda" and torch.cuda.is_available():
            # CUDA optimizations
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("CUDA: Enabled deterministic mode")
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                logger.info("CUDA: Enabled benchmark mode for performance")
            
            # Memory management
            torch.cuda.empty_cache()
            logger.info("CUDA: Cleared cache")
        
        # CPU optimizations
        if device.type == "cpu":
            # Set number of threads for CPU operations
            available_cores = torch.get_num_threads()
            if available_cores > 4:
                torch.set_num_threads(available_cores)
                logger.info(f"CPU: Using {Available_cores} threads")
    
    def clear_cache(self) -> None:
        """Clear device cache and free GPU memory if available."""
        self._device_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")


# Global device manager instance
device_manager = DeviceManager()