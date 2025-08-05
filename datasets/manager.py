"""High-level dataset management with SVD integration."""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging

from core import ExperimentConfig, SVDFeatureReducer, get_logger
from .base import DatasetInfo, DataSplitter, DatasetProcessor, create_data_loaders, validate_cross_domain_compatibility
from .loaders import GraphDatasetLoader, create_dataset_loader

logger = get_logger(__name__)


class DatasetManager:
    """High-level manager for dataset loading, processing, and SVD alignment."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize dataset manager.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Initialize components
        self.loader = create_dataset_loader('graph', 
                                           normalize_features=True)
        self.splitter = DataSplitter(
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            shuffle=config.shuffle,
            random_state=config.seed
        )
        self.processor = DatasetProcessor()
        
        # SVD reducer cache
        self._svd_reducers = {}
        
        # Dataset cache
        self._dataset_cache = {}
    
    def load_single_domain_dataset(self, dataset_name: Optional[str] = None) -> Tuple[DatasetInfo, DataLoader, DataLoader, DataLoader, Optional[SVDFeatureReducer]]:
        """Load dataset for single domain experiments.
        
        Args:
            dataset_name: Dataset name (uses config if None)
            
        Returns:
            Tuple of (dataset_info, train_loader, val_loader, test_loader, svd_reducer)
        """
        if dataset_name is None:
            dataset_name = self.config.source_dataset
        
        if not dataset_name:
            raise ValueError("Dataset name not specified in config or arguments")
        
        logger.info(f"Loading single domain dataset: {dataset_name}")
        
        # Load raw dataset
        data, dataset_info = self.loader.load_raw_dataset(dataset_name)
        
        # Create splits
        data = self.splitter.create_splits(data)
        
        # Apply SVD reduction if enabled
        svd_reducer = None
        if self.config.feature_reduction_enable:
            data, dataset_info, svd_reducer = self._apply_svd_reduction(
                data, dataset_info, dataset_name, fit_new=True
            )
        
        # Process dataset
        data, dataset_info = self.processor.process(data, dataset_info)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(data)
        
        logger.info(f"✅ Single domain dataset loaded: {dataset_info}")
        return dataset_info, train_loader, val_loader, test_loader, svd_reducer
    
    def load_cross_domain_datasets(self, source_name: Optional[str] = None, 
                                  target_name: Optional[str] = None) -> Tuple[DatasetInfo, DatasetInfo, DataLoader, DataLoader, DataLoader, DataLoader, Optional[SVDFeatureReducer]]:
        """Load datasets for cross-domain experiments with SVD alignment.
        
        Args:
            source_name: Source dataset name (uses config if None)
            target_name: Target dataset name (uses config if None)
            
        Returns:
            Tuple of (source_info, target_info, source_loader, target_train_loader, 
                     target_val_loader, target_test_loader, source_svd_reducer)
        """
        if source_name is None:
            source_name = self.config.source_dataset
        if target_name is None:
            target_name = self.config.target_dataset
        
        if not source_name or not target_name:
            raise ValueError("Source and target dataset names must be specified")
        
        logger.info(f"Loading cross-domain datasets: {source_name} → {target_name}")
        
        # Load source dataset
        source_data, source_info = self.loader.load_raw_dataset(source_name)
        
        # Load target dataset
        target_data, target_info = self.loader.load_raw_dataset(target_name)
        
        # Create splits for target dataset
        target_data = self.splitter.create_splits(target_data)
        
        # Apply SVD reduction if enabled
        source_svd_reducer = None
        if self.config.feature_reduction_enable:
            # Apply SVD to source dataset (fit new or load existing)
            source_data, source_info, source_svd_reducer = self._apply_svd_reduction(
                source_data, source_info, source_name, fit_new=True
            )
            
            # Apply same SVD transformation to target dataset
            target_data, target_info = self._apply_existing_svd_reduction(
                target_data, target_info, source_svd_reducer
            )
            
            logger.info("✅ SVD alignment completed between source and target")
        
        # Validate cross-domain compatibility
        validate_cross_domain_compatibility(source_info, target_info, strict=True)
        
        # Process datasets
        source_data, source_info = self.processor.process(source_data, source_info)
        target_data, target_info = self.processor.process(target_data, target_info)
        
        # Create data loaders
        source_loader = DataLoader([source_data], batch_size=1)
        target_train_loader, target_val_loader, target_test_loader = create_data_loaders(target_data)
        
        logger.info(f"✅ Cross-domain datasets loaded:")
        logger.info(f"   Source: {source_info}")
        logger.info(f"   Target: {target_info}")
        
        return (source_info, target_info, source_loader, 
                target_train_loader, target_val_loader, target_test_loader, 
                source_svd_reducer)
    
    def _apply_svd_reduction(self, data: Data, dataset_info: DatasetInfo, 
                            dataset_name: str, fit_new: bool = False) -> Tuple[Data, DatasetInfo, SVDFeatureReducer]:
        """Apply SVD reduction to dataset features.
        
        Args:
            data: Graph data
            dataset_info: Dataset information
            dataset_name: Name of the dataset
            fit_new: Whether to fit new SVD or load existing
            
        Returns:
            Tuple of (reduced_data, updated_info, svd_reducer)
        """
        target_dim = self.config.feature_reduction_target_dim
        svd_path = Path("checkpoints") / f"{dataset_name}_svd_reducer.pkl"
        
        logger.info(f"Applying SVD reduction: {data.x.size(1)}D → {target_dim}D")
        
        # Load existing or create new SVD reducer
        if not fit_new and svd_path.exists():
            logger.info(f"Loading existing SVD reducer: {svd_path}")
            svd_reducer = SVDFeatureReducer.load(svd_path)
        else:
            logger.info("Creating new SVD reducer")
            svd_reducer = SVDFeatureReducer(
                target_dim=target_dim,
                random_state=self.config.seed
            )
            svd_reducer.fit(data.x)
            
            # Save SVD reducer if requested
            if self.config.feature_reduction_save_reducer:
                svd_reducer.save(svd_path)
        
        # Transform features
        original_features = data.x.clone()
        data.x = svd_reducer.transform(data.x)
        
        # Update dataset info
        dataset_info.original_num_features = original_features.size(1)
        dataset_info.num_features = data.x.size(1)
        dataset_info.svd_applied = True
        dataset_info.svd_info = svd_reducer.get_info()
        
        logger.info(f"✅ SVD reduction applied. Explained variance: {svd_reducer.explained_variance_ratio.sum():.4f}")
        
        # Cache reducer
        self._svd_reducers[dataset_name] = svd_reducer
        
        return data, dataset_info, svd_reducer
    
    def _apply_existing_svd_reduction(self, data: Data, dataset_info: DatasetInfo, 
                                    svd_reducer: SVDFeatureReducer) -> Tuple[Data, DatasetInfo]:
        """Apply existing SVD reduction to align with source dataset.
        
        Args:
            data: Graph data
            dataset_info: Dataset information
            svd_reducer: Fitted SVD reducer from source dataset
            
        Returns:
            Tuple of (aligned_data, updated_info)
        """
        logger.info(f"Applying source SVD to target dataset: {data.x.size(1)}D → {svd_reducer.target_dim}D")
        
        # Store original features
        original_features = data.x.clone()
        
        # Apply SVD transformation
        data.x = svd_reducer.transform(data.x)
        
        # Update dataset info
        dataset_info.original_num_features = original_features.size(1)
        dataset_info.num_features = data.x.size(1)
        dataset_info.svd_applied = True
        dataset_info.svd_info = svd_reducer.get_info()
        
        return data, dataset_info
    
    def get_svd_reducer(self, dataset_name: str) -> Optional[SVDFeatureReducer]:
        """Get cached SVD reducer for a dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            SVD reducer if available, None otherwise
        """
        return self._svd_reducers.get(dataset_name)
    
    def load_dataset_by_config(self) -> Union[
        Tuple[DatasetInfo, DataLoader, DataLoader, DataLoader, Optional[SVDFeatureReducer]],
        Tuple[DatasetInfo, DatasetInfo, DataLoader, DataLoader, DataLoader, DataLoader, Optional[SVDFeatureReducer]]
    ]:
        """Load dataset(s) based on experiment configuration.
        
        Returns:
            Dataset loaders based on experiment type
        """
        if self.config.type == 'single_domain':
            return self.load_single_domain_dataset()
        elif self.config.type == 'cross_domain':
            return self.load_cross_domain_datasets()
        else:
            raise ValueError(f"Unknown experiment type: {self.config.type}")
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Get dataset information without loading the full dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dataset information
        """
        # Check cache first
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name][1]
        
        # Load dataset info
        _, dataset_info = self.loader.load_raw_dataset(dataset_name)
        
        return dataset_info
    
    def clear_cache(self):
        """Clear all caches."""
        self.loader.clear_cache()
        self._dataset_cache.clear()
        self._svd_reducers.clear()
        logger.info("All dataset caches cleared")
    
    def get_supported_datasets(self) -> list:
        """Get list of supported datasets."""
        return self.loader.get_supported_datasets()
    
    def validate_dataset_names(self) -> bool:
        """Validate dataset names in configuration.
        
        Returns:
            True if all dataset names are valid
            
        Raises:
            ValueError: If any dataset name is invalid
        """
        supported = self.get_supported_datasets()
        
        if self.config.type == 'single_domain':
            if self.config.source_dataset not in supported:
                raise ValueError(f"Unknown dataset: {self.config.source_dataset}. "
                               f"Supported: {supported}")
        elif self.config.type == 'cross_domain':
            if self.config.source_dataset not in supported:
                raise ValueError(f"Unknown source dataset: {self.config.source_dataset}. "
                               f"Supported: {supported}")
            if self.config.target_dataset not in supported:
                raise ValueError(f"Unknown target dataset: {self.config.target_dataset}. "
                               f"Supported: {supported}")
        
        return True


# Convenience function for backward compatibility
def load_dataset(config: ExperimentConfig) -> Union[
    Tuple[DatasetInfo, DataLoader, DataLoader, DataLoader, Optional[SVDFeatureReducer]],
    Tuple[DatasetInfo, DatasetInfo, DataLoader, DataLoader, DataLoader, DataLoader, Optional[SVDFeatureReducer]]
]:
    """Load dataset(s) based on configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dataset loaders based on experiment type
    """
    manager = DatasetManager(config)
    return manager.load_dataset_by_config()