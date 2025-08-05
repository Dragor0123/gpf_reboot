"""Dataset loading and management for GPF experiments."""

from .base import (
    DatasetInfo,
    BaseDatasetLoader,
    DataSplitter,
    DatasetProcessor,
    create_data_loaders,
    get_dataset_statistics,
    validate_cross_domain_compatibility
)
from .loaders import (
    GraphDatasetLoader,
    CustomDatasetLoader,
    create_dataset_loader,
    get_all_supported_datasets,
    validate_dataset_availability,
    load_citation_network,
    load_amazon_dataset
)
from .manager import DatasetManager, load_dataset

__all__ = [
    'DatasetInfo',
    'BaseDatasetLoader',
    'DataSplitter',
    'DatasetProcessor',
    'create_data_loaders',
    'get_dataset_statistics',
    'validate_cross_domain_compatibility',
    'GraphDatasetLoader',
    'CustomDatasetLoader',
    'create_dataset_loader',
    'get_all_supported_datasets',
    'validate_dataset_availability',
    'load_citation_network',
    'load_amazon_dataset',
    'DatasetManager',
    'load_dataset',
]