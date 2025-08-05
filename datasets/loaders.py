"""Concrete dataset loaders for various graph datasets."""

import torch
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from typing import Dict, Any, Tuple, List
from pathlib import Path
import logging

from .base import BaseDatasetLoader, DatasetInfo

logger = logging.getLogger(__name__)


class GraphDatasetLoader(BaseDatasetLoader):
    """Loader for common graph datasets (Planetoid, Amazon, etc.)."""
    
    # Dataset name mappings for convenience
    DATASET_MAPPINGS = {
        'computers': 'amazoncomputers',
        'computer': 'amazoncomputers', 
        'photo': 'amazonphoto',
        'amazonphoto': 'amazonphoto',
        'amazoncomputers': 'amazoncomputers',
    }
    
    def __init__(self, root_dir: str = './datasets', cache_enabled: bool = True,
                 normalize_features: bool = True):
        """Initialize graph dataset loader.
        
        Args:
            root_dir: Root directory for dataset storage
            cache_enabled: Whether to cache loaded datasets
            normalize_features: Whether to normalize node features
        """
        super().__init__(root_dir, cache_enabled)
        self.normalize_features = normalize_features
        
        # Create transform pipeline
        self.transform = NormalizeFeatures() if normalize_features else None
    
    def get_supported_datasets(self) -> List[str]:
        """Get list of supported dataset names."""
        return [
            'cora', 'citeseer', 'pubmed',  # Planetoid
            'amazoncomputers', 'computers', 'computer',  # Amazon Computers (with aliases)
            'amazonphoto', 'photo',  # Amazon Photo (with aliases)
            'actor',  # Actor
            'chameleon', 'squirrel'  # Wikipedia networks
        ]
    
    def load_raw_dataset(self, name: str) -> Tuple[Data, DatasetInfo]:
        """Load raw dataset from PyTorch Geometric.
        
        Args:
            name: Dataset name
            
        Returns:
            Tuple of (graph_data, dataset_info)
        """
        name = self.validate_dataset_name(name)
        
        # Check cache first
        cache_key = self.get_cache_key(name, normalize=self.normalize_features)
        if self.cache_enabled and cache_key in self._dataset_cache:
            logger.info(f"Loading {name} from cache")
            return self._dataset_cache[cache_key]
        
        logger.info(f"Loading {name} dataset...")
        
        # Map alias to canonical name
        canonical_name = self.DATASET_MAPPINGS.get(name, name)
        
        # Load dataset using appropriate loader
        dataset = self._load_pyg_dataset(canonical_name)
        data = dataset[0]  # Get the first (and usually only) graph
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=name,
            num_features=data.x.size(1),
            num_classes=dataset.num_classes,
            num_nodes=data.num_nodes,
            num_edges=data.num_edges,
            original_num_features=data.x.size(1),  # Will be updated by SVD if applied
            svd_applied=False,
            dataset_type=self._get_dataset_type(canonical_name),
            is_directed=self._is_directed_graph(data),
            has_node_features=hasattr(data, 'x') and data.x is not None,
            has_edge_features=hasattr(data, 'edge_attr') and data.edge_attr is not None,
        )
        
        # Cache result
        if self.cache_enabled:
            self._dataset_cache[cache_key] = (data, dataset_info)
        
        logger.info(f"✅ Loaded {name}: {dataset_info}")
        return data, dataset_info
    
    def _load_pyg_dataset(self, name: str):
        """Load dataset using PyTorch Geometric loaders."""
        root = self.root_dir / name
        
        # Planetoid datasets
        if name in ['cora', 'citeseer', 'pubmed']:
            return Planetoid(
                root=str(root), 
                name=name.capitalize(), 
                transform=self.transform
            )
        
        # Amazon datasets
        elif name == 'amazonphoto':
            return Amazon(
                root=str(root), 
                name='Photo', 
                transform=self.transform
            )
        elif name == 'amazoncomputers':
            return Amazon(
                root=str(root), 
                name='Computers', 
                transform=self.transform
            )
        
        # Actor dataset
        elif name == 'actor':
            return Actor(root=str(root), transform=self.transform)
        
        # Wikipedia networks
        elif name in ['chameleon', 'squirrel']:
            return WikipediaNetwork(
                root=str(root), 
                name=name.capitalize(), 
                transform=self.transform
            )
        
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    def _get_dataset_type(self, name: str) -> str:
        """Get dataset type/family."""
        if name in ['cora', 'citeseer', 'pubmed']:
            return 'citation'
        elif name in ['amazonphoto', 'amazoncomputers']:
            return 'amazon'
        elif name == 'actor':
            return 'actor'
        elif name in ['chameleon', 'squirrel']:
            return 'wikipedia'
        else:
            return 'unknown'
    
    def _is_directed_graph(self, data: Data) -> bool:
        """Check if graph is directed by examining edge reciprocity."""
        if not hasattr(data, 'edge_index') or data.edge_index.size(1) == 0:
            return False
        
        # Check if all edges have reverse edges
        edge_index = data.edge_index
        edge_set = set()
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))
        
        # Check reciprocity
        reciprocal_count = 0
        for src, dst in edge_set:
            if (dst, src) in edge_set:
                reciprocal_count += 1
        
        reciprocity = reciprocal_count / len(edge_set)
        return reciprocity < 0.9  # Consider directed if reciprocity < 90%
    
    def get_dataset_metadata(self, name: str) -> Dict[str, Any]:
        """Get comprehensive metadata for a dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Dictionary with dataset metadata
        """
        name = self.validate_dataset_name(name)
        canonical_name = self.DATASET_MAPPINGS.get(name, name)
        
        # Load dataset to get actual statistics
        data, info = self.load_raw_dataset(name)
        
        # Compute additional statistics
        from .base import get_dataset_statistics
        stats = get_dataset_statistics(data)
        
        # Combine info and stats
        metadata = info.to_dict()
        metadata.update(stats)
        
        # Add dataset-specific information
        metadata.update({
            'canonical_name': canonical_name,
            'dataset_family': self._get_dataset_type(canonical_name),
            'loader_class': self.__class__.__name__,
            'supports_inductive': canonical_name in ['amazoncomputers', 'amazonphoto'],
            'citation_network': canonical_name in ['cora', 'citeseer', 'pubmed'],
        })
        
        return metadata


class CustomDatasetLoader(BaseDatasetLoader):
    """Loader for custom datasets in various formats."""
    
    def __init__(self, root_dir: str = './datasets', cache_enabled: bool = True):
        super().__init__(root_dir, cache_enabled)
        self.custom_datasets = {}
    
    def register_dataset(self, name: str, loader_func, **metadata):
        """Register a custom dataset loader function.
        
        Args:
            name: Dataset name
            loader_func: Function that returns (Data, DatasetInfo)
            **metadata: Additional metadata for the dataset
        """
        self.custom_datasets[name] = {
            'loader': loader_func,
            'metadata': metadata
        }
        logger.info(f"Registered custom dataset: {name}")
    
    def get_supported_datasets(self) -> List[str]:
        """Get list of registered custom datasets."""
        return list(self.custom_datasets.keys())
    
    def load_raw_dataset(self, name: str) -> Tuple[Data, DatasetInfo]:
        """Load custom dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Tuple of (graph_data, dataset_info)
        """
        name = self.validate_dataset_name(name)
        
        # Check cache
        cache_key = self.get_cache_key(name)
        if self.cache_enabled and cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        
        # Load using registered loader
        dataset_config = self.custom_datasets[name]
        loader_func = dataset_config['loader']
        
        logger.info(f"Loading custom dataset: {name}")
        data, dataset_info = loader_func()
        
        # Add registered metadata
        for key, value in dataset_config['metadata'].items():
            dataset_info.metadata[key] = value
        
        # Cache result
        if self.cache_enabled:
            self._dataset_cache[cache_key] = (data, dataset_info)
        
        return data, dataset_info


def create_dataset_loader(loader_type: str = 'graph', **kwargs) -> BaseDatasetLoader:
    """Factory function to create dataset loaders.
    
    Args:
        loader_type: Type of loader ('graph', 'custom')
        **kwargs: Arguments for the loader
        
    Returns:
        Dataset loader instance
    """
    loaders = {
        'graph': GraphDatasetLoader,
        'custom': CustomDatasetLoader,
    }
    
    if loader_type not in loaders:
        raise ValueError(f"Unknown loader type: {loader_type}. "
                        f"Available: {list(loaders.keys())}")
    
    loader_class = loaders[loader_type]
    return loader_class(**kwargs)


def get_all_supported_datasets() -> Dict[str, List[str]]:
    """Get all supported datasets organized by loader type.
    
    Returns:
        Dictionary mapping loader types to supported datasets
    """
    result = {}
    
    # Graph datasets
    graph_loader = GraphDatasetLoader()
    result['graph'] = graph_loader.get_supported_datasets()
    
    return result


def validate_dataset_availability(dataset_name: str) -> Tuple[bool, str]:
    """Check if a dataset is available and return its loader type.
    
    Args:
        dataset_name: Name of the dataset to check
        
    Returns:
        Tuple of (is_available, loader_type)
    """
    all_datasets = get_all_supported_datasets()
    
    for loader_type, datasets in all_datasets.items():
        if dataset_name.lower() in [d.lower() for d in datasets]:
            return True, loader_type
    
    return False, ""


# Convenience functions for common use cases
def load_citation_network(name: str, root_dir: str = './datasets', 
                         normalize_features: bool = True) -> Tuple[Data, DatasetInfo]:
    """Load a citation network (Cora, CiteSeer, PubMed).
    
    Args:
        name: Citation network name
        root_dir: Root directory for datasets
        normalize_features: Whether to normalize features
        
    Returns:
        Tuple of (graph_data, dataset_info)
    """
    if name.lower() not in ['cora', 'citeseer', 'pubmed']:
        raise ValueError(f"Unknown citation network: {name}")
    
    loader = GraphDatasetLoader(root_dir=root_dir, normalize_features=normalize_features)
    return loader.load_raw_dataset(name)


def load_amazon_dataset(name: str, root_dir: str = './datasets',
                       normalize_features: bool = True) -> Tuple[Data, DatasetInfo]:
    """Load an Amazon dataset (Computers, Photo).
    
    Args:
        name: Amazon dataset name ('computers', 'photo')
        root_dir: Root directory for datasets
        normalize_features: Whether to normalize features
        
    Returns:
        Tuple of (graph_data, dataset_info)
    """
    if name.lower() not in ['computers', 'computer', 'photo', 'amazoncomputers', 'amazonphoto']:
        raise ValueError(f"Unknown Amazon dataset: {name}")
    
    loader = GraphDatasetLoader(root_dir=root_dir, normalize_features=normalize_features)
    return loader.load_raw_dataset(name)