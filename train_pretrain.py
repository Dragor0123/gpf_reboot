import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from pathlib import Path

from utils import (
    set_seed, get_device, load_config,
    setup_logging, log_experiment_info,
    save_ckpt, create_run_name
)
from models import create_model
from models.encoders import ProjectionHead
from datasets import load_dataset
from torch_geometric.data import Data


def graph_views(data: Data, aug: str = 'dropN', aug_ratio: float = 0.2):
    """
    Create augmented graph views for contrastive learning.
    
    Args:
        data: Original graph data
        aug: Augmentation type
        aug_ratio: Augmentation ratio
    
    Returns:
        Augmented graph data
    """
    x, edge_index = data.x.clone(), data.edge_index.clone()

    if aug == 'dropN':  # Node feature masking
        mask = torch.rand(x.size(0), device=x.device) > aug_ratio
        x = x * mask.unsqueeze(1).float()

    elif aug == 'permE':  # Edge permutation (randomly drop edges)
        num_edges = edge_index.size(1)
        keep = torch.rand(num_edges, device=edge_index.device) > aug_ratio
        edge_index = edge_index[:, keep]

    elif aug == 'maskN':  # Feature masking (random noise)
        noise = torch.randn_like(x) * aug_ratio
        x = x + noise

    elif aug == 'dropE':  # Explicitly drop edges
        num_edges = edge_index.size(1)
        drop = torch.rand(num_edges, device=edge_index.device) < aug_ratio
        edge_index = edge_index[:, ~drop]

    elif aug == 'maskF':  # Hard masking of random features
        mask = torch.rand_like(x) > aug_ratio
        x = x * mask

    return Data(x=x, edge_index=edge_index)


def info_nce_loss(z1, z2, temperature=0.5):
    """
    InfoNCE contrastive loss.
    
    Args:
        z1: First view embeddings [N, D]
        z2: Second view embeddings [N, D]
        temperature: Temperature parameter
    
    Returns:
        InfoNCE loss (scalar)
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    # Positive pairs are on the diagonal
    positives = torch.diag(sim_matrix)
    
    # Numerator: exp(positive similarities)
    numerator = torch.exp(positives)
    
    # Denominator: sum of exp(all similarities) for each row
    denominator = torch.sum(torch.exp(sim_matrix), dim=1)
    
    # InfoNCE loss
    loss = -torch.log(numerator / denominator)
    
    return loss.mean()


def pretrain_single_domain(config):
    """
    Pretrain on single domain dataset with SVD.
    """
    logging.info("Starting single domain pretraining with SVD...")
    
    # Load dataset with SVD reduction
    dataset_info, train_loader, _, _, svd_reducer = load_dataset(config)
    data = next(iter(train_loader))
    
    return pretrain_on_data(config, data, dataset_info, "single_domain", svd_reducer)


def pretrain_cross_domain(config):
    """
    Pretrain on source dataset for cross-domain transfer with SVD.
    """
    logging.info("Starting cross-domain pretraining on source dataset with SVD...")
    
    # Load cross-domain datasets with SVD alignment
    (source_info, target_info, source_loader, _, _, _, 
     source_svd_reducer) = load_dataset(config)
    
    if source_loader is None:
        raise ValueError("Source loader is None - check source_free configuration")
    
    source_data = next(iter(source_loader))
    
    return pretrain_on_data(config, source_data, source_info, "cross_domain", source_svd_reducer)


def pretrain_on_data(config, data, dataset_info, experiment_type, svd_reducer=None):
    """
    Core pretraining logic with SVD-reduced features.
    
    Args:
        config: Configuration dictionary
        data: Graph data to pretrain on (SVD-reduced)
        dataset_info: Dataset information
        experiment_type: Type of experiment
        svd_reducer: SVD reducer used for this dataset
    
    Returns:
        Trained encoder
    """
    device = get_device(config['experiment']['device'])
    data = data.to(device)
    
    # Log SVD information
    if svd_reducer and dataset_info['svd_applied']:
        logging.info(f"🔧 Pre-training with SVD-reduced features:")
        logging.info(f"   Original dimension: {dataset_info['original_num_features']}")
        logging.info(f"   Reduced dimension: {dataset_info['num_features']}")
        logging.info(f"   Explained variance: {dataset_info['svd_info']['explained_variance_ratio']:.4f}")
    
    # Model + Projection Head (using SVD-reduced dimension)
    encoder = create_model(
        model_type=config['model']['type'],
        input_dim=dataset_info['num_features'],  # SVD-reduced dimension
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    projector = ProjectionHead(
        input_dim=config['model']['hidden_dim'],
        hidden_dim=config['model']['hidden_dim']
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=config['pretrain']['lr'],
        weight_decay=config['pretrain']['weight_decay']
    )

    # Augmentation settings
    aug_config = config['pretrain']['augmentation']
    aug1 = aug_config['view1']
    aug2 = aug_config['view2']
    aug_ratio = aug_config['aug_ratio']
    temperature = aug_config['temperature']

    logging.info("Starting GCL-style contrastive pretraining...")
    logging.info(f"Dataset: {dataset_info['dataset_name']}")
    logging.info(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    logging.info(f"Features: {dataset_info['num_features']} (SVD-reduced)")
    
    # Training loop
    encoder.train()
    projector.train()
    
    for epoch in range(config['pretrain']['epochs']):
        # Create augmented views
        view1 = graph_views(data, aug=aug1, aug_ratio=aug_ratio)
        view2 = graph_views(data, aug=aug2, aug_ratio=aug_ratio)

        # Forward pass
        h1 = encoder(view1.x.to(device), view1.edge_index.to(device))
        h2 = encoder(view2.x.to(device), view2.edge_index.to(device))

        # Project to contrastive space
        z1 = projector(h1)
        z2 = projector(h2)

        # Compute contrastive loss
        loss = info_nce_loss(z1, z2, temperature=temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            logging.info(f"Epoch {epoch:04d} | Contrastive Loss: {loss:.4f}")

    # Save final encoder with source dataset name and SVD info
    source_dataset = config['experiment'].get('source_dataset', config['dataset'].get('name', 'unknown'))
    save_path = f"checkpoints/{source_dataset}_encoder_final.pt"
    
    # Include SVD information in checkpoint
    checkpoint_info = {
        'dataset_info': dataset_info,
        'experiment_type': experiment_type,
        'config': config,
        'svd_reducer_path': f"checkpoints/{source_dataset}_svd_reducer.pkl" if svd_reducer else None
    }
    
    save_ckpt(
        encoder, optimizer, epoch, loss.item(), save_path,
        **checkpoint_info
    )
    
    logging.info(f"Pretraining complete. Model saved to: {save_path}")
    if svd_reducer:
        logging.info(f"✅ SVD reducer saved to: checkpoints/{source_dataset}_svd_reducer.pkl")
    
    return encoder


def main():
    """
    Main pretraining function.
    """
    # Load configuration
    config = load_config("config.yaml")
    set_seed(config['experiment']['seed'])
    device = get_device(config['experiment']['device'])
    setup_logging(config['experiment']['log_level'])
    log_experiment_info(config)
    
    # Create run name
    run_name = create_run_name(config)
    config['experiment']['run_name'] = run_name
    
    logging.info(f"Starting pretraining for run: {run_name}")
    
    # Determine experiment type and run pretraining
    experiment_type = config['experiment']['type']
    
    try:
        if experiment_type == 'single_domain':
            encoder = pretrain_single_domain(config)
        elif experiment_type == 'cross_domain':
            # 사전훈련 단계에서는 source 접근이 허용되어야 함
            encoder = pretrain_cross_domain(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        logging.info("✅ Pretraining completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Pretraining failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()