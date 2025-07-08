import torch
import logging
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

from utils import SVDFeatureReducer


def _load_single_dataset(name: str, root_dir: str = './datasets'):
    """
    Load a single dataset by name.
    
    Args:
        name: Dataset name
        root_dir: Root directory for datasets
        
    Returns:
        PyG dataset object
    """
    name = name.lower()
    root = Path(root_dir) / name
    
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=str(root), name=name.capitalize(), transform=NormalizeFeatures())
    elif name in ['amazonphoto', 'photo']:
        dataset = Amazon(root=str(root), name='Photo', transform=NormalizeFeatures())
    elif name in ['amazoncomputers', 'computers', 'computer']:
        dataset = Amazon(root=str(root), name='Computers', transform=NormalizeFeatures())
    elif name == 'actor':
        dataset = Actor(root=str(root), transform=NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=str(root), name=name.capitalize(), transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    
    return dataset


def _create_splits(data, val_ratio: float = 0.1, test_ratio: float = 0.2, shuffle: bool = True):
    """
    Create train/val/test splits for a dataset.
    
    Args:
        data: PyG data object
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        shuffle: Whether to shuffle indices
        
    Returns:
        Modified data object with masks
    """
    num_nodes = data.num_nodes
    
    # Generate indices
    if shuffle:
        indices = torch.randperm(num_nodes)
    else:
        indices = torch.arange(num_nodes)
    
    # Calculate split sizes
    val_size = int(num_nodes * val_ratio)
    test_size = int(num_nodes * test_ratio)
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
    
    return data


def load_single_domain_dataset(config: Dict[str, Any]):
    """
    Load dataset for single domain experiments with optional SVD reduction.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (dataset_info, train_loader, val_loader, test_loader, svd_reducer)
    """
    dataset_name = config['dataset']['name']
    split_config = config['dataset']['split']
    
    logging.info(f"Loading single domain dataset: {dataset_name}")
    
    # Load dataset
    dataset = _load_single_dataset(dataset_name)
    data = dataset[0]
    
    # Create splits
    data = _create_splits(
        data,
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio'],
        shuffle=split_config['shuffle']
    )
    
    # Apply SVD reduction if enabled
    svd_reducer = None
    original_features = data.x.clone()
    
    if config['feature_reduction']['enable']:
        target_dim = config['feature_reduction']['target_dim']
        
        logging.info(f"🔧 Applying SVD reduction: {data.x.size(1)}D → {target_dim}D")
        
        # Create and fit SVD reducer
        svd_reducer = SVDFeatureReducer(target_dim=target_dim)
        data.x = svd_reducer.fit_transform(data.x)
        
        # Save SVD reducer
        if config['feature_reduction']['save_reducer']:
            svd_path = f"checkpoints/{dataset_name}_svd_reducer.pkl"
            svd_reducer.save(svd_path)
        
        logging.info(f"✅ SVD applied. Explained variance: {svd_reducer.explained_variance_ratio.sum():.4f}")
    
    # Create loaders
    train_loader = DataLoader([data], batch_size=1)
    val_loader = DataLoader([data], batch_size=1)
    test_loader = DataLoader([data], batch_size=1)
    
    # Dataset info
    dataset_info = {
        'num_features': data.x.size(1),  # SVD reduced dimension
        'original_num_features': original_features.size(1),
        'num_classes': dataset.num_classes,
        'dataset_name': dataset_name,
        'svd_applied': config['feature_reduction']['enable'],
        'svd_info': svd_reducer.get_info() if svd_reducer else None
    }
    
    logging.info(f"Dataset info: {dataset_info}")
    return dataset_info, train_loader, val_loader, test_loader, svd_reducer


def load_cross_domain_datasets(config: Dict[str, Any]):
    """
    Load datasets for cross-domain experiments with SVD alignment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (source_info, target_info, source_loader, target_train_loader, 
                 target_val_loader, target_test_loader, source_svd_reducer)
    """
    source_name = config['experiment']['source_dataset']
    target_name = config['experiment']['target_dataset']
    split_config = config['dataset']['split']
    
    logging.info(f"Loading cross-domain datasets:")
    logging.info(f"  Source: {source_name}")
    logging.info(f"  Target: {target_name}")
    
    # Load source dataset
    source_dataset = _load_single_dataset(source_name)
    source_data = source_dataset[0]
    
    # Load target dataset
    target_dataset = _load_single_dataset(target_name)
    target_data = target_dataset[0]
    
    # Create splits for target dataset
    target_data = _create_splits(
        target_data,
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio'],
        shuffle=split_config['shuffle']
    )
    
    # Store original dimensions
    source_original_dim = source_data.x.size(1)
    target_original_dim = target_data.x.size(1)
    
    # Apply SVD reduction if enabled
    source_svd_reducer = None
    
    if config['feature_reduction']['enable']:
        target_dim = config['feature_reduction']['target_dim']
        
        logging.info(f"🔧 Applying SVD reduction to source: {source_original_dim}D → {target_dim}D")
        
        # Load existing SVD reducer for source or create new one
        source_svd_path = f"checkpoints/{source_name}_svd_reducer.pkl"
        
        if Path(source_svd_path).exists():
            logging.info(f"📂 Loading existing SVD reducer: {source_svd_path}")
            source_svd_reducer = SVDFeatureReducer.load(source_svd_path)
            source_data.x = source_svd_reducer.transform(source_data.x)
        else:
            logging.info(f"🔧 Creating new SVD reducer for source")
            source_svd_reducer = SVDFeatureReducer(target_dim=target_dim)
            source_data.x = source_svd_reducer.fit_transform(source_data.x)
            
            # Save SVD reducer
            if config['feature_reduction']['save_reducer']:
                source_svd_reducer.save(source_svd_path)
        
        # Apply same SVD transformation to target data
        logging.info(f"🔧 Applying source SVD to target: {target_original_dim}D → {target_dim}D")
        target_data.x = source_svd_reducer.transform(target_data.x)
        
        logging.info(f"✅ SVD applied to both datasets")
        logging.info(f"   Source explained variance: {source_svd_reducer.explained_variance_ratio.sum():.4f}")
    
    # Check feature dimension compatibility after SVD
    if source_data.x.size(1) != target_data.x.size(1):
        logging.warning(f"Feature dimension mismatch after SVD: "
                       f"source={source_data.x.size(1)}, "
                       f"target={target_data.x.size(1)}")
    else:
        logging.info(f"✅ Feature dimensions aligned: {source_data.x.size(1)}D")
    
    # Create loaders
    source_loader = DataLoader([source_data], batch_size=1)
    target_train_loader = DataLoader([target_data], batch_size=1)
    target_val_loader = DataLoader([target_data], batch_size=1)
    target_test_loader = DataLoader([target_data], batch_size=1)
    
    # Dataset info
    source_info = {
        'num_features': source_data.x.size(1),  # SVD reduced dimension
        'original_num_features': source_original_dim,
        'num_classes': source_dataset.num_classes,
        'dataset_name': source_name,
        'num_nodes': source_data.num_nodes,
        'svd_applied': config['feature_reduction']['enable'],
        'svd_info': source_svd_reducer.get_info() if source_svd_reducer else None
    }
    
    target_info = {
        'num_features': target_data.x.size(1),  # SVD reduced dimension (same as source)
        'original_num_features': target_original_dim,
        'num_classes': target_dataset.num_classes,
        'dataset_name': target_name,
        'num_nodes': target_data.num_nodes,
        'svd_applied': config['feature_reduction']['enable'],
        'svd_info': source_svd_reducer.get_info() if source_svd_reducer else None
    }
    
    logging.info(f"Source dataset info: {source_info}")
    logging.info(f"Target dataset info: {target_info}")
    
    return (source_info, target_info, source_loader, target_train_loader, 
            target_val_loader, target_test_loader, source_svd_reducer)


def load_dataset(config: Dict[str, Any]):
    """
    Load datasets based on experiment type.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dataset information and loaders (format depends on experiment type)
    """
    experiment_type = config['experiment']['type']
    
    if experiment_type == 'single_domain':
        return load_single_domain_dataset(config)
    elif experiment_type == 'cross_domain':
        return load_cross_domain_datasets(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def get_dataset_statistics(data):
    """
    Get basic statistics of a dataset.
    
    Args:
        data: PyG data object
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.x.size(1),
        'avg_degree': float(data.num_edges) / data.num_nodes * 2,  # undirected graph
        'density': float(data.num_edges) / (data.num_nodes * (data.num_nodes - 1)) * 2
    }
    
    if hasattr(data, 'y'):
        stats['num_classes'] = int(data.y.max()) + 1
        stats['class_distribution'] = torch.bincount(data.y).tolist()
    
    return stats


def validate_cross_domain_compatibility(source_info: Dict, target_info: Dict) -> bool:
    """
    Validate if source and target datasets are compatible for cross-domain experiments.
    
    Args:
        source_info: Source dataset information
        target_info: Target dataset information
        
    Returns:
        True if compatible
    """
    # After SVD, feature dimensions should match
    if source_info['num_features'] != target_info['num_features']:
        logging.error(f"Feature dimension mismatch after SVD: "
                     f"source={source_info['num_features']}, "
                     f"target={target_info['num_features']}")
        return False
    
    # Class 수가 다르면 warning만
    if source_info['num_classes'] != target_info['num_classes']:
        logging.warning(f"Number of classes differs: "
                       f"source={source_info['num_classes']}, "
                       f"target={target_info['num_classes']}")
    
    # SVD 적용 상태 확인
    if source_info['svd_applied'] != target_info['svd_applied']:
        logging.warning(f"SVD application mismatch: "
                       f"source={source_info['svd_applied']}, "
                       f"target={target_info['svd_applied']}")
    
    logging.info(f"✅ Cross-domain compatibility validated")
    return True


def get_svd_reducer_path(dataset_name: str) -> str:
    """
    Get SVD reducer file path for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to SVD reducer file
    """
    return f"checkpoints/{dataset_name}_svd_reducer.pkl"


def check_svd_reducer_exists(dataset_name: str) -> bool:
    """
    Check if SVD reducer exists for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        True if SVD reducer file exists
    """
    return Path(get_svd_reducer_path(dataset_name)).exists()


# Legacy function for backward compatibility
def load_dataset_legacy(name, batch_size=128, val_ratio=0.1, test_ratio=0.2, shuffle=True):
    """
    Legacy function for backward compatibility.
    """
    logging.warning("Using legacy load_dataset function. Consider updating to new API.")
    
    # Create minimal config
    config = {
        'experiment': {'type': 'single_domain'},
        'dataset': {
            'name': name,
            'split': {
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'shuffle': shuffle
            }
        },
        'feature_reduction': {
            'enable': False  # Disable SVD for legacy
        }
    }
    
    return load_single_domain_dataset(config)