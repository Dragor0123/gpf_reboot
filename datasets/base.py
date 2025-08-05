"""Base classes for dataset loading and management."""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetInfo:
    """Container for dataset metadata and statistics."""
    
    def __init__(self, name: str, num_features: int, num_classes: int, 
                 num_nodes: int = 0, num_edges: int = 0, **kwargs):
        self.name = name
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        
        # Store additional metadata
        self.metadata = kwargs
        
        # SVD-related information
        self.original_num_features = kwargs.get('original_num_features', num_features)
        self.svd_applied = kwargs.get('svd_applied', False)
        self.svd_info = kwargs.get('svd_info', None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'original_num_features': self.original_num_features,
            'svd_applied': self.svd_applied,
            'svd_info': self.svd_info,
            **self.metadata
        }
    
    def __repr__(self):
        return (f"DatasetInfo(name='{self.name}', features={self.num_features}, "
                f"classes={self.num_classes}, nodes={self.num_nodes})")


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, root_dir: str = './datasets', cache_enabled: bool = True):
        self.root_dir = Path(root_dir)
        self.cache_enabled = cache_enabled
        self._dataset_cache = {}
    
    @abstractmethod
    def load_raw_dataset(self, name: str) -> Tuple[Data, DatasetInfo]:
        """Load raw dataset from storage.
        
        Args:
            name: Dataset name
            
        Returns:
            Tuple of (data, dataset_info)
        """
        pass
    
    @abstractmethod
    def get_supported_datasets(self) -> List[str]:
        """Get list of supported dataset names."""
        pass
    
    def validate_dataset_name(self, name: str) -> str:
        """Validate and normalize dataset name.
        
        Args:
            name: Dataset name to validate
            
        Returns:
            Normalized dataset name
            
        Raises:
            ValueError: If dataset is not supported
        """
        name = name.lower()
        supported = self.get_supported_datasets()
        
        if name not in supported:
            raise ValueError(f"Unsupported dataset: {name}. "
                           f"Supported datasets: {', '.join(supported)}")
        
        return name
    
    def get_cache_key(self, name: str, **kwargs) -> str:
        """Generate cache key for dataset."""
        key_parts = [name]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "_".join(key_parts)
    
    def clear_cache(self):
        """Clear dataset cache."""
        self._dataset_cache.clear()
        logger.info("Dataset cache cleared")


class DataSplitter:
    """Handles train/validation/test splits for graph datasets."""
    
    def __init__(self, val_ratio: float = 0.1, test_ratio: float = 0.2, 
                 shuffle: bool = True, random_state: int = 42):
        """Initialize data splitter.
        
        Args:
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            shuffle: Whether to shuffle node indices
            random_state: Random seed for reproducible splits
        """
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.random_state = random_state
        
        # Validate ratios
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        if val_ratio + test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be less than 1")
    
    def create_splits(self, data: Data, 
                     stratify: bool = False) -> Data:
        """Create train/val/test splits for graph data.
        
        Args:
            data: Graph data object
            stratify: Whether to stratify splits by class (if labels available)
            
        Returns:
            Data object with mask attributes added
        """
        num_nodes = data.num_nodes
        
        # Generate indices
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.random_state)
            indices = torch.randperm(num_nodes, generator=generator)
        else:
            indices = torch.arange(num_nodes)
        
        # Handle stratified splits if requested and labels are available
        if stratify and hasattr(data, 'y') and data.y is not None:
            indices = self._stratified_split(indices, data.y)
        
        # Calculate split sizes
        val_size = int(num_nodes * self.val_ratio)
        test_size = int(num_nodes * self.test_ratio)
        train_size = num_nodes - val_size - test_size
        
        # Split indices
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Create masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        
        # Log split information
        logger.info(f"Created splits: train={train_size}, val={val_size}, test={test_size}")
        
        # Log class distribution if available
        if hasattr(data, 'y') and data.y is not None:
            self._log_class_distribution(data)
        
        return data
    
    def _stratified_split(self, indices: torch.Tensor, 
                         labels: torch.Tensor) -> torch.Tensor:
        """Create stratified splits maintaining class proportions."""
        # Get unique classes and their indices
        unique_classes = torch.unique(labels)
        stratified_indices = []
        
        for class_label in unique_classes:
            class_mask = labels[indices] == class_label
            class_indices = indices[class_mask]
            
            if len(class_indices) > 0:
                stratified_indices.append(class_indices)
        
        # Interleave indices to maintain rough class balance
        result_indices = []
        max_class_size = max(len(ci) for ci in stratified_indices)
        
        for i in range(max_class_size):
            for class_indices in stratified_indices:
                if i < len(class_indices):
                    result_indices.append(class_indices[i])
        
        return torch.tensor(result_indices)
    
    def _log_class_distribution(self, data: Data):
        """Log class distribution for each split."""
        def get_class_dist(mask):
            if mask.sum() == 0:
                return {}
            labels = data.y[mask]
            unique, counts = torch.unique(labels, return_counts=True)
            return {int(c): int(count) for c, count in zip(unique, counts)}
        
        train_dist = get_class_dist(data.train_mask)
        val_dist = get_class_dist(data.val_mask)
        test_dist = get_class_dist(data.test_mask)
        
        logger.info("Class distribution:")
        logger.info(f"  Train: {train_dist}")
        logger.info(f"  Val: {val_dist}")
        logger.info(f"  Test: {test_dist}")


class DatasetProcessor:
    """Handles dataset preprocessing and transformations."""
    
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, transform_func):
        """Add a transformation function to the pipeline."""
        self.transformations.append(transform_func)
    
    def process(self, data: Data, dataset_info: DatasetInfo) -> Tuple[Data, DatasetInfo]:
        """Apply all transformations to the data.
        
        Args:
            data: Graph data
            dataset_info: Dataset information
            
        Returns:
            Processed data and updated info
        """
        processed_data = data
        processed_info = dataset_info
        
        for transform in self.transformations:
            processed_data, processed_info = transform(processed_data, processed_info)
        
        return processed_data, processed_info


def create_data_loaders(data: Data, batch_size: int = 1, 
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train/val/test splits.
    
    Args:
        data: Graph data with mask attributes
        batch_size: Batch size (typically 1 for node classification)
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # For node classification, we typically use the full graph
    # The masks determine which nodes are used for each split
    train_loader = DataLoader([data], batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers)
    val_loader = DataLoader([data], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader([data], batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def get_dataset_statistics(data: Data) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics.
    
    Args:
        data: Graph data object
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.x.size(1) if hasattr(data, 'x') and data.x is not None else 0,
    }
    
    # Graph-level statistics
    if data.num_edges > 0:
        stats['avg_degree'] = float(data.num_edges) / data.num_nodes * 2  # undirected
        stats['graph_density'] = float(data.num_edges) / (data.num_nodes * (data.num_nodes - 1)) * 2
    else:
        stats['avg_degree'] = 0.0
        stats['graph_density'] = 0.0
    
    # Node features statistics
    if hasattr(data, 'x') and data.x is not None:
        stats['feature_dim'] = data.x.size(1)
        stats['feature_mean'] = data.x.mean().item()
        stats['feature_std'] = data.x.std().item()
        stats['feature_min'] = data.x.min().item()
        stats['feature_max'] = data.x.max().item()
    
    # Label statistics
    if hasattr(data, 'y') and data.y is not None:
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        stats['num_classes'] = len(unique_labels)
        stats['class_distribution'] = {
            int(label): int(count) for label, count in zip(unique_labels, counts)
        }
        stats['label_balance'] = float(counts.min()) / float(counts.max())
    
    # Split statistics (if available)
    for split_name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(data, split_name):
            mask = getattr(data, split_name)
            stats[f'{split_name}_size'] = int(mask.sum())
            stats[f'{split_name}_ratio'] = float(mask.sum()) / data.num_nodes
    
    return stats


def validate_cross_domain_compatibility(source_info: DatasetInfo, 
                                       target_info: DatasetInfo,
                                       strict: bool = True) -> bool:
    """Validate compatibility between source and target datasets.
    
    Args:
        source_info: Source dataset information
        target_info: Target dataset information
        strict: Whether to enforce strict compatibility checks
        
    Returns:
        True if compatible
        
    Raises:
        ValueError: If datasets are incompatible and strict=True
    """
    issues = []
    
    # Feature dimension compatibility (critical)
    if source_info.num_features != target_info.num_features:
        msg = (f"Feature dimension mismatch: source={source_info.num_features}, "
               f"target={target_info.num_features}")
        if strict:
            raise ValueError(msg)
        else:
            issues.append(msg)
    
    # Class count compatibility (warning)
    if source_info.num_classes != target_info.num_classes:
        msg = (f"Number of classes differs: source={source_info.num_classes}, "
               f"target={target_info.num_classes}")
        issues.append(msg)
    
    # SVD compatibility check
    if source_info.svd_applied != target_info.svd_applied:
        msg = (f"SVD application mismatch: source={source_info.svd_applied}, "
               f"target={target_info.svd_applied}")
        if strict:
            raise ValueError(msg)
        else:
            issues.append(msg)
    
    # Log warnings for non-critical issues
    for issue in issues:
        if strict:
            logger.error(issue)
        else:
            logger.warning(issue)
    
    if not issues:
        logger.info("✅ Cross-domain compatibility validated")
    
    return len(issues) == 0 or not strict