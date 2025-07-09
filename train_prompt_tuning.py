import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

from utils import (
    set_seed, get_device, load_config,
    setup_logging, log_experiment_info,
    load_ckpt, EarlyStopping,
    save_results, create_run_name, print_model_info
)
from models import create_model, Classifier
from datasets import load_dataset, validate_cross_domain_compatibility
from prompts.gpf_prompt import GPFPrompt, ResidualMLPPrompt
from losses import TargetCentricLoss
from anchor_factory import generate_gaussian_anchors


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
    device = get_device(config['experiment']['device'])
    data = data.to(device)
    
    # Log SVD information
    if svd_reducer and dataset_info['svd_applied']:
        logging.info(f"🔧 Prompt tuning with SVD-aligned features:")
        logging.info(f"   Original target dimension: {dataset_info['original_num_features']}")
        logging.info(f"   Aligned dimension: {dataset_info['num_features']}")
        logging.info(f"   SVD explained variance: {dataset_info['svd_info']['explained_variance_ratio']:.4f}")
    
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
        expected_input_dim = checkpoint_dataset_info.get('num_features', dataset_info['num_features'])
        
        if expected_input_dim != dataset_info['num_features']:
            raise ValueError(f"❌ Dimension mismatch after SVD! "
                           f"Expected {expected_input_dim}, got {dataset_info['num_features']}. "
                           f"Check SVD alignment.")
        
        logging.info(f"✅ Dimension verification passed: {expected_input_dim}D")
    
    # Create encoder with aligned dimension
    encoder = create_model(
        model_type=config['model']['type'],
        input_dim=dataset_info['num_features'],  # SVD-aligned dimension
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

    # Initialize prompt (dimensions already aligned)
    prompt = GPFPrompt(
        input_dim=dataset_info['num_features'],  # SVD-aligned dimension
        p_num=config['prompt']['num_prompts']
    ).to(device)
    
    # Initialize classifier
    classifier = Classifier(
        input_dim=config['model']['hidden_dim'],
        num_classes=dataset_info['num_classes']
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

    # Initialize target-centric loss
    loss_fn = TargetCentricLoss(config).to(device)
    
    # Initialize regularizer if target-centric is enabled
    if config['target_centric']['enable']:
        logging.info("🎯 Target-Centric Prior Modeling enabled")

        anchor_cfg = config['target_centric']['regularization']['anchor']
        if anchor_cfg['type'] == 'gaussian':
            latent_dim = config['model']['hidden_dim']
            num_anchors = anchor_cfg['num_anchors']
            anchors = generate_gaussian_anchors(num_anchors, latent_dim, device)
            loss_fn.regularizer.initialize_fixed_anchors(anchors)
        else:
            # 일반적인 target feature 기반 정렬
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

        # Compute loss
        loss_dict = loss_fn(
            logits=logits,
            labels=y,
            embeddings=h,
            mask=train_mask,
            edge_index=edge_index
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
            if config['target_centric']['enable']:
                log_msg += f"Reg Loss: {reg_loss:.4f} | "
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
        'dataset': dataset_info['dataset_name'],
        'target_centric_enabled': config['target_centric']['enable'],
        'svd_applied': dataset_info['svd_applied'],
        'svd_info': dataset_info.get('svd_info'),
        'original_target_dim': dataset_info.get('original_num_features'),
        'aligned_dim': dataset_info['num_features'],
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
    if dataset_info['svd_applied']:
        logging.info(f"SVD DIM REDUCTION: {dataset_info.get('original_num_features')}D → {dataset_info['num_features']}D")
        logging.info(f"EXPLAINED VARIANCE: {dataset_info['svd_info']['explained_variance_ratio']:.4f}")
    logging.info("="*50)
    
    return final_results


def main():
    """
    Main prompt tuning function.
    """
    # Load configuration
    config = load_config("config.yaml")
    set_seed(config['experiment']['seed'])
    device = get_device(config['experiment']['device'])
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