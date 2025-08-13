import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
import argparse

# Import from refactored modules
from core.config import ConfigManager
from core.device import device_manager
from core.logging import setup_logging, get_logger
from core.reproducibility import set_reproducible_seeds
from models import create_model, Classifier, print_model_info
from datasets import DatasetManager, validate_cross_domain_compatibility
from prompts.prompt_function import GPFPrompt, ResidualMLPPrompt, LinearPrompt
from training.losses import TargetCentricLoss
from anchor_factory import generate_gaussian_anchors, generate_mog_anchors, generate_mog_anchors_simple
import yaml
import datetime
import os


def evaluate_model(encoder, prompt, classifier, data, mask, device, return_embeddings=False):
    """
    Evaluate model performance.
    
    Args:
        encoder: Frozen encoder
        prompt: Prompt module
        classifier: Classification head
        data: Graph data
        mask: Evaluation mask
        device: Device
        return_embeddings: Whether to return embeddings
    
    Returns:
        Evaluation metrics (and optionally embeddings)
    """
    encoder.eval()
    prompt.eval()
    classifier.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    with torch.no_grad():
        # Apply prompt
        prompted_x = prompt.add(x)
        
        # Get embeddings
        h = encoder(prompted_x, edge_index)
        
        # Get predictions
        logits = classifier(h)
        preds = logits[mask].argmax(dim=1).cpu()
        probs = F.softmax(logits[mask], dim=1).cpu()
        labels = y[mask].cpu()

    # Compute metrics
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')

    # AUROC computation
    try:
        if logits.size(1) == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels, probs, multi_class='ovo')
    except ValueError:
        auroc = 0.0  # Handle edge cases

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'auroc': auroc
    }

    if return_embeddings:
        return metrics, h
    return metrics


def train_single_domain(config):
    """
    Train on single domain dataset with SVD.
    """
    logging.info("Starting single domain prompt tuning with SVD...")
    
    # Load dataset with SVD reduction
    dataset_info, train_loader, val_loader, test_loader, svd_reducer = load_dataset(config)
    data = next(iter(train_loader))
    
    return train_on_target_data(config, data, dataset_info, "single_domain", svd_reducer)


def train_cross_domain(config):
    """
    Train on target dataset for cross-domain transfer with SVD alignment.
    """
    logging.info("Starting cross-domain prompt tuning on target dataset with SVD...")
    
    # Load cross-domain datasets with SVD alignment
    (source_info, target_info, source_loader, target_train_loader, 
     target_val_loader, target_test_loader, source_svd_reducer) = load_dataset(config)
    
    # Validate compatibility
    validate_cross_domain_compatibility(source_info, target_info)
    
    target_data = next(iter(target_train_loader))
    
    return train_on_target_data(config, target_data, target_info, "cross_domain", source_svd_reducer)


def create_prompt(config, input_dim):
    """Create prompt based on configuration.
    
    Args:
        config: Configuration dictionary
        input_dim: Input feature dimension
        
    Returns:
        Prompt module
    """
    prompt_config = config['prompt']
    prompt_type = prompt_config['type'].lower()
    
    if prompt_type == 'gpf':
        gpf_config = prompt_config['gpf']
        prompt = GPFPrompt(
            input_dim=input_dim,
            p_num=gpf_config['num_prompts']
        )
        logging.info(f"Created GPFPrompt with {gpf_config['num_prompts']} prompts")
        
    elif prompt_type == 'residual_mlp':
        mlp_config = prompt_config['residual_mlp']
        prompt = ResidualMLPPrompt(
            input_dim=input_dim,
            hidden_dim=mlp_config['hidden_dim'],
            num_layers=mlp_config['num_layers'],
            dropout=mlp_config['dropout']
        )
        logging.info(f"Created ResidualMLPPrompt with hidden_dim={mlp_config['hidden_dim']}, "
                    f"num_layers={mlp_config['num_layers']}, dropout={mlp_config['dropout']}")
        
    elif prompt_type == 'linear':
        linear_config = prompt_config['linear']
        out_channels = linear_config.get('out_channels', None)
        prompt = LinearPrompt(
            in_channels=input_dim,
            out_channels=out_channels
        )
        effective_out_channels = out_channels if out_channels is not None else input_dim
        logging.info(f"Created LinearPrompt with in_channels={input_dim}, out_channels={effective_out_channels}")
        
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Supported types: 'gpf', 'residual_mlp', 'linear'")
    
    return prompt


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    set_reproducible_seeds(seed)


def log_experiment_info(config):
    """Log experiment configuration."""
    logger = get_logger(__name__)
    logger.info("=" * 50)
    logger.info("🚀 EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Experiment Type: {config['experiment']['type']}")
    if config['experiment']['type'] == 'cross_domain':
        logger.info(f"Source Dataset: {config['experiment']['source_dataset']}")
        logger.info(f"Target Dataset: {config['experiment']['target_dataset']}")
    else:
        logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Model: {config['model']['type']}")
    logger.info(f"Prompt Type: {config['prompt']['type']}")
    logger.info(f"Target-Centric: {config['target_centric']['enable']}")
    logger.info(f"SVD Reduction: {config['feature_reduction']['enable']}")
    logger.info("=" * 50)


def create_run_name(config):
    """Create a unique run name based on configuration."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if config['experiment']['type'] == 'cross_domain':
        source = config['experiment']['source_dataset']
        target = config['experiment']['target_dataset']
        name_parts = [f"{source}_to_{target}"]
    else:
        name_parts = [config['dataset']['name']]
    
    # Add SVD info
    if config['feature_reduction']['enable']:
        name_parts.append(f"svd{config['feature_reduction']['target_dim']}")
    
    # Add prompt type
    name_parts.append(config['prompt']['type'])
    
    # Add target-centric info
    if config['target_centric']['enable']:
        reg_type = config['target_centric']['regularization']['divergence']['type']
        beta = config['target_centric']['regularization']['beta']
        name_parts.append(f"tc_{reg_type}_{beta}")
    else:
        name_parts.append("baseline")
    
    name_parts.append(timestamp)
    
    return "_".join(name_parts)


def save_results(results, config):
    """Save experiment results to file."""
    results_dir = Path(config['evaluation']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory based on experiment type
    if config['feature_reduction']['enable']:
        subdir = "svd_applied"
    else:
        subdir = "no_svd"
    
    if config['target_centric']['enable']:
        anchor_type = config['target_centric']['regularization']['anchor']['type']
        if anchor_type in ['gaussian', 'mog']:
            subdir = os.path.join(subdir, f"{anchor_type}_anchor")
        else:
            subdir = os.path.join(subdir, "target_centric")
    else:
        subdir = os.path.join(subdir, "base")
    
    save_dir = results_dir / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = config['experiment']['run_name']
    results_file = save_dir / f"{run_name}_results.yaml"
    
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger = get_logger(__name__)
    logger.info(f"💾 Results saved to: {results_file}")


def load_dataset(config):
    """Load dataset based on experiment configuration."""
    # Create a simplified config object that works with DatasetManager
    from core.config import ConfigManager
    config_manager = ConfigManager()
    experiment_config = config_manager.load_config("config.yaml")
    
    dataset_manager = DatasetManager(experiment_config)
    
    if config['experiment']['type'] == 'single_domain':
        return dataset_manager.load_single_domain_dataset()
    else:
        return dataset_manager.load_cross_domain_datasets()


def load_ckpt(checkpoint_path, model, device='cpu', strict=True):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    logger = get_logger(__name__)
    logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_on_target_data(config, data, dataset_info, experiment_type, svd_reducer=None):
    """
    Core prompt tuning logic with SVD-aligned features.
    
    Args:
        config: Configuration dictionary
        data: Target graph data (SVD-aligned)
        dataset_info: Target dataset information
        experiment_type: Type of experiment
        svd_reducer: SVD reducer used for alignment
    
    Returns:
        Training results
    """
    device = device_manager.get_device(config['experiment']['device'])
    data = data.to(device)
    
    # Log SVD information
    if svd_reducer and dataset_info.svd_applied:
        logging.info(f"🔧 Prompt tuning with SVD-aligned features:")
        logging.info(f"   Original target dimension: {dataset_info.original_num_features}")
        logging.info(f"   Aligned dimension: {dataset_info.num_features}")
        svd_info = getattr(dataset_info, 'svd_info', None)
        if svd_info:
            logging.info(f"   SVD explained variance: {svd_info['explained_variance_ratio']:.4f}")
    
    # Load pretrained encoder (frozen) - dimensions should now match perfectly!
    if experiment_type == 'cross_domain':
        source_dataset = config['experiment']['source_dataset']
    else:
        source_dataset = config['dataset']['name']
    
    checkpoint_path = f"checkpoints/{source_dataset}_encoder_final.pt"
    
    # Load checkpoint to verify SVD alignment
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_dataset_info = checkpoint.get('dataset_info', {})
        expected_input_dim = checkpoint_dataset_info.get('num_features', dataset_info.num_features)
        
        if expected_input_dim != dataset_info.num_features:
            raise ValueError(f"❌ Dimension mismatch after SVD! "
                           f"Expected {expected_input_dim}, got {dataset_info.num_features}. "
                           f"Check SVD alignment.")
        
        logging.info(f"✅ Dimension verification passed: {expected_input_dim}D")
    
    # Create encoder with aligned dimension
    encoder = create_model(
        model_type=config['model']['type'],
        input_dim=dataset_info.num_features,  # SVD-aligned dimension
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Pretrained model not found: {checkpoint_path}")
    
    # Load with strict=True (dimensions should match perfectly now!)
    load_ckpt(checkpoint_path, encoder, device=device, strict=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    logging.info("✅ Loaded pretrained encoder (frozen) with perfect dimension alignment")
    print_model_info(encoder, "Encoder")

    # Initialize prompt based on configuration
    prompt = create_prompt(config, dataset_info.num_features).to(device)
    
    # Initialize classifier
    classifier = Classifier(
        input_dim=config['model']['hidden_dim'],
        num_classes=dataset_info.num_classes
    ).to(device)
    
    print_model_info(prompt, "Prompt")
    print_model_info(classifier, "Classifier")

    # Optimizer
    cfg = config['prompt_tuning']
    optimizer = torch.optim.Adam(
        list(prompt.parameters()) + list(classifier.parameters()),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )

    # Early stopping
    early_stopping = None
    if cfg.get('early_stopping', {}).get('enable', False):
        early_stopping = EarlyStopping(
            patience=cfg['early_stopping']['patience'],
            min_delta=cfg['early_stopping']['min_delta'],
            mode='max'
        )

    # Initialize target-centric loss with flattened config for compatibility
    loss_config = {
        'target_centric_enable': config['target_centric']['enable'],
        'target_centric_beta': config['target_centric']['regularization']['beta'],
        'target_centric_anchor_type': config['target_centric']['regularization']['anchor']['type'],
        'target_centric_anchor_num_anchors': config['target_centric']['regularization']['anchor']['num_anchors'],
        'target_centric_anchor_num_components': config['target_centric']['regularization']['anchor']['num_components'],
        'target_centric_anchor_use_sklearn_gmm': config['target_centric']['regularization']['anchor']['use_sklearn_gmm'],
        'target_centric_mapper_type': config['target_centric']['regularization']['mapper']['type'],
        'target_centric_divergence_type': config['target_centric']['regularization']['divergence']['type'],
        'target_centric_divergence_sigma': config['target_centric']['regularization']['divergence']['params']['sigma'],
    }
    loss_fn = TargetCentricLoss(loss_config).to(device)
    
    # Initialize regularizer if target-centric is enabled
    if config['target_centric']['enable']:
        logging.info("🎯 Target-Centric Prior Modeling enabled")
        anchor_cfg = config['target_centric']['regularization']['anchor']
        anchor_type = anchor_cfg['type']
        
        if anchor_type == 'gaussian':
            latent_dim = config['model']['hidden_dim']
            num_anchors = anchor_cfg['num_anchors']
            anchors = generate_gaussian_anchors(num_anchors, latent_dim, device)
            loss_fn.regularizer.initialize_fixed_anchors(anchors)
        elif anchor_type == 'mog':
            num_components = anchor_cfg.get('num_components', 5)
            num_anchors = anchor_cfg.get('num_anchors', 500)  # 추가!
            
            with torch.no_grad():
                encoder.eval()
                latent_z = encoder(data.x, data.edge_index)  # edge_index 추가
            
            # Choose implementation
            use_sklearn_gmm = anchor_cfg.get('use_sklearn_gmm', False)
            
            if use_sklearn_gmm:
                anchors = generate_mog_anchors(latent_z, num_components, num_anchors)
            else:
                anchors = generate_mog_anchors_simple(latent_z, num_components, num_anchors)
            
            loss_fn.regularizer.initialize_fixed_anchors(anchors)
            logging.info(f"🎯 MoG anchors initialized: {anchors.shape}")
        else:
            # 기존 방식: feature-based anchor 선택 + mapper
            target_features = data.x
            edge_index = data.edge_index if hasattr(data, 'edge_index') else None
            loss_fn.initialize_regularizer_with_target_features(
                target_features, encoder, edge_index
            )
    
    # Training data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    logging.info(f"Training data: {train_mask.sum()} nodes")
    logging.info(f"Validation data: {val_mask.sum()} nodes")
    logging.info(f"Test data: {test_mask.sum()} nodes")

    # Training loop
    best_val_score = 0.0
    results = {'train_losses': [], 'val_metrics': [], 'epoch_details': []}
    
    for epoch in range(cfg['epochs']):
        # Training
        prompt.train()
        classifier.train()

        # Forward pass
        prompted_x = prompt.add(x)
        h = encoder(prompted_x, edge_index)
        logits = classifier(h)

        # Compute loss (with epoch for dynamic anchor updates)
        loss_dict = loss_fn(
            logits=logits,
            labels=y,
            embeddings=h,
            mask=train_mask,
            edge_index=edge_index,
            epoch=epoch
        )

        total_loss = loss_dict['total_loss']
        task_loss = loss_dict['task_loss']
        reg_loss = loss_dict['reg_loss']
        
        # NaN detection and early exit
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error(f"❌ NaN/Inf detected at epoch {epoch}!")
            logging.error(f"   Task Loss: {task_loss.item()}")
            logging.error(f"   Reg Loss: {reg_loss.item()}")
            logging.error(f"   This should not happen with SVD alignment!")
            break

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record training loss
        results['train_losses'].append({
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'reg_loss': reg_loss.item()
        })

        # Validation
        if epoch % 10 == 0 or epoch == cfg['epochs'] - 1:
            val_metrics = evaluate_model(encoder, prompt, classifier, data, val_mask, device)
            results['val_metrics'].append({
                'epoch': epoch,
                **val_metrics
            })
            
            val_score = val_metrics['accuracy']
            
            # Log progress
            log_msg = f"Epoch {epoch:03d} | "
            log_msg += f"Total Loss: {total_loss:.4f} | "
            log_msg += f"Task Loss: {task_loss:.4f} | "
            
            # Add regularization loss info
            if config.get('dynamic_anchor', {}).get('enable', False) or config['target_centric']['enable']:
                log_msg += f"Reg Loss: {reg_loss:.4f} | "
                
                # Add dynamic anchor specific info
                if 'anchor_update_info' in loss_dict:
                    update_info = loss_dict['anchor_update_info']
                    if update_info['anchors_updated']:
                        log_msg += f"Anchors Updated | "
                    if update_info['using_fallback']:
                        log_msg += f"Fallback Active | "
            
            log_msg += f"Val Acc: {val_score:.4f}"
            logging.info(log_msg)
            
            # Track best model
            if val_score > best_val_score:
                best_val_score = val_score
                
                # Save best model state
                best_state = {
                    'prompt': prompt.state_dict(),
                    'classifier': classifier.state_dict(),
                    'epoch': epoch,
                    'val_score': val_score
                }

            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_score):
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break

    # Final evaluation on test set
    if 'best_state' in locals():
        prompt.load_state_dict(best_state['prompt'])
        classifier.load_state_dict(best_state['classifier'])
    
    test_metrics = evaluate_model(encoder, prompt, classifier, data, test_mask, device)
    
    # Final results
    final_results = {
        'experiment_type': experiment_type,
        'dataset': dataset_info.name,
        'target_centric_enabled': config['target_centric']['enable'],
        'svd_applied': dataset_info.svd_applied,
        'svd_info': getattr(dataset_info, 'svd_info', None),
        'original_target_dim': dataset_info.original_num_features,
        'aligned_dim': dataset_info.num_features,
        'best_val_score': best_val_score,
        'test_metrics': test_metrics,
        'training_history': results,
        'config': config
    }
    
    # Log final results
    logging.info("\n" + "="*50)
    logging.info("📊 FINAL TEST RESULTS")
    logging.info("="*50)
    for metric, value in test_metrics.items():
        logging.info(f"{metric.upper():<12}: {value:.4f}")
    if dataset_info.svd_applied:
        logging.info(f"SVD DIM REDUCTION: {dataset_info.original_num_features}D → {dataset_info.num_features}D")
        svd_info = getattr(dataset_info, 'svd_info', None)
        if svd_info:
            logging.info(f"EXPLAINED VARIANCE: {svd_info['explained_variance_ratio']:.4f}")
    logging.info("="*50)
    
    return final_results


def main():
    """
    Main prompt tuning function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Graph Prompt Tuning')
    parser.add_argument('--source_dataset', type=str, help='Source dataset name')
    parser.add_argument('--target_dataset', type=str, help='Target dataset name')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.source_dataset:
        config['experiment']['source_dataset'] = args.source_dataset
    if args.target_dataset:
        config['experiment']['target_dataset'] = args.target_dataset
    if args.seed:
        config['experiment']['seed'] = args.seed
    
    set_seed(config['experiment']['seed'])
    device = device_manager.get_device(config['experiment']['device'])
    setup_logging(config['experiment']['log_level'])
    log_experiment_info(config)
    
    # Create or use existing run name
    if 'run_name' not in config['experiment'] or config['experiment']['run_name'] is None:
        run_name = create_run_name(config)
        config['experiment']['run_name'] = run_name
    
    logging.info(f"Starting prompt tuning for run: {config['experiment']['run_name']}")
    
    # Determine experiment type and run training
    experiment_type = config['experiment']['type']
    
    try:
        if experiment_type == 'single_domain':
            results = train_single_domain(config)
        elif experiment_type == 'cross_domain':
            results = train_cross_domain(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Save results
        if config['evaluation']['save_results']:
            save_results(results, config)
        
        logging.info("✅ Prompt tuning completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"❌ Prompt tuning failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()