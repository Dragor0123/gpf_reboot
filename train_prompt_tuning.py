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


def debug_tensor(tensor, name, check_stats=True):
    """강력한 텐서 디버깅 함수"""
    if tensor is None:
        logging.error(f"❌ {name}: None")
        return False
    
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    logging.info(f"🔍 {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    if has_nan:
        logging.error(f"❌ {name}: Contains NaN!")
        nan_count = torch.isnan(tensor).sum().item()
        logging.error(f"   NaN count: {nan_count}/{tensor.numel()}")
        return False
    
    if has_inf:
        logging.error(f"❌ {name}: Contains Inf!")
        inf_count = torch.isinf(tensor).sum().item()
        logging.error(f"   Inf count: {inf_count}/{tensor.numel()}")
        return False
    
    if check_stats:
        try:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            
            logging.info(f"   Stats: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
            
            # 극단적인 값 체크
            if abs(min_val) > 1000 or abs(max_val) > 1000:
                logging.warning(f"⚠️  {name}: Extreme values detected!")
                
        except Exception as e:
            logging.error(f"❌ {name}: Error computing stats: {e}")
            return False
    
    return True


def safe_forward_pass(encoder, prompt, classifier, x, edge_index, step_name=""):
    """안전한 forward pass with 단계별 디버깅"""
    logging.info(f"🔍 Safe forward pass: {step_name}")
    
    # Step 1: Input check
    if not debug_tensor(x, "Input features"):
        return None, None, None
    
    # Step 2: Prompt application
    try:
        prompted_x = prompt.add(x)
        if not debug_tensor(prompted_x, "Prompted features"):
            return None, None, None
    except Exception as e:
        logging.error(f"❌ Prompt forward failed: {e}")
        return None, None, None
    
    # Step 3: Encoder forward
    try:
        with torch.no_grad():
            encoder.eval()
            h = encoder(prompted_x, edge_index)
        if not debug_tensor(h, "Encoder embeddings"):
            return None, None, None
    except Exception as e:
        logging.error(f"❌ Encoder forward failed: {e}")
        return None, None, None
    
    # Step 4: Classifier forward
    try:
        logits = classifier(h)
        if not debug_tensor(logits, "Classifier logits"):
            return None, None, None
    except Exception as e:
        logging.error(f"❌ Classifier forward failed: {e}")
        return None, None, None
    
    logging.info(f"✅ Safe forward pass completed: {step_name}")
    return prompted_x, h, logits


def evaluate_model(encoder, prompt, classifier, data, mask, device, return_embeddings=False):
    """
    Evaluate model performance with safety checks.
    """
    encoder.eval()
    prompt.eval()
    classifier.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    with torch.no_grad():
        # Safe forward pass
        prompted_x, h, logits = safe_forward_pass(
            encoder, prompt, classifier, x, edge_index, "Evaluation"
        )
        
        if logits is None:
            logging.error("❌ Evaluation failed - returning zero metrics")
            return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0, 'auroc': 0.0}
        
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
    Core prompt tuning logic with extensive debugging.
    """
    device = get_device(config['experiment']['device'])
    data = data.to(device)
    
    logging.info("="*80)
    logging.info("🔍 DEBUGGING INFORMATION")
    logging.info("="*80)
    
    # Debug data
    debug_tensor(data.x, "Raw target data", check_stats=True)
    debug_tensor(data.edge_index, "Edge index", check_stats=False)
    if hasattr(data, 'y'):
        debug_tensor(data.y, "Labels", check_stats=False)
    
    # Log SVD information
    if svd_reducer and dataset_info['svd_applied']:
        logging.info(f"🔧 SVD Information:")
        logging.info(f"   Original target dimension: {dataset_info['original_num_features']}")
        logging.info(f"   Aligned dimension: {dataset_info['num_features']}")
        logging.info(f"   SVD explained variance: {dataset_info['svd_info']['explained_variance_ratio']:.4f}")
    
    # Load pretrained encoder
    if experiment_type == 'cross_domain':
        source_dataset = config['experiment']['source_dataset']
    else:
        source_dataset = config['dataset']['name']
    
    checkpoint_path = f"checkpoints/{source_dataset}_encoder_final.pt"
    logging.info(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Detailed checkpoint verification
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_dataset_info = checkpoint.get('dataset_info', {})
        expected_input_dim = checkpoint_dataset_info.get('num_features', dataset_info['num_features'])
        
        logging.info(f"🔍 Checkpoint verification:")
        logging.info(f"   Expected input dim: {expected_input_dim}")
        logging.info(f"   Actual target dim: {dataset_info['num_features']}")
        
        if expected_input_dim != dataset_info['num_features']:
            logging.error(f"❌ CRITICAL: Dimension mismatch after SVD!")
            logging.error(f"   This should not happen with proper SVD alignment!")
            logging.error(f"   Expected: {expected_input_dim}, Got: {dataset_info['num_features']}")
            raise ValueError("Dimension mismatch despite SVD alignment")
        
        logging.info(f"✅ Dimension verification passed: {expected_input_dim}D")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create encoder with verified dimension
    logging.info("🏗️  Creating encoder...")
    encoder = create_model(
        model_type=config['model']['type'],
        input_dim=dataset_info['num_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load checkpoint with strict=True
    logging.info("📥 Loading encoder weights...")
    try:
        load_ckpt(checkpoint_path, encoder, device=device, strict=True)
        logging.info("✅ Encoder weights loaded successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load encoder weights: {e}")
        raise
    
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    logging.info("✅ Encoder frozen successfully")
    print_model_info(encoder, "Encoder")

    # Test encoder immediately with a small batch
    logging.info("🧪 Testing encoder with sample data...")
    with torch.no_grad():
        sample_x = data.x[:10]  # Small sample
        sample_edge_index = data.edge_index[:, :20]  # Small edges
        
        if not debug_tensor(sample_x, "Sample input"):
            raise ValueError("Sample input contains NaN/Inf")
        
        try:
            sample_h = encoder(sample_x, sample_edge_index)
            if not debug_tensor(sample_h, "Sample encoder output"):
                raise ValueError("Encoder output contains NaN/Inf")
            logging.info("✅ Encoder test passed")
        except Exception as e:
            logging.error(f"❌ Encoder test failed: {e}")
            raise

    # Initialize prompt
    logging.info("🏗️  Creating prompt...")
    prompt = GPFPrompt(
        input_dim=dataset_info['num_features'],
        p_num=config['prompt']['num_prompts']
    ).to(device)
    
    # Test prompt immediately
    logging.info("🧪 Testing prompt...")
    with torch.no_grad():
        try:
            sample_prompted = prompt.add(sample_x)
            if not debug_tensor(sample_prompted, "Sample prompt output"):
                raise ValueError("Prompt output contains NaN/Inf")
            logging.info("✅ Prompt test passed")
        except Exception as e:
            logging.error(f"❌ Prompt test failed: {e}")
            raise
    
    # Initialize classifier
    logging.info("🏗️  Creating classifier...")
    classifier = Classifier(
        input_dim=config['model']['hidden_dim'],
        num_classes=dataset_info['num_classes']
    ).to(device)
    
    print_model_info(prompt, "Prompt")
    print_model_info(classifier, "Classifier")

    # Full pipeline test before training
    logging.info("🧪 Full pipeline test...")
    prompted_x, h, logits = safe_forward_pass(
        encoder, prompt, classifier, sample_x, sample_edge_index, "Pre-training test"
    )
    
    if logits is None:
        logging.error("❌ Pipeline test failed - aborting training")
        raise ValueError("Pipeline test failed")
    
    logging.info("✅ Full pipeline test passed")

    # Initialize target-centric loss
    logging.info("🏗️  Initializing loss function...")
    loss_fn = TargetCentricLoss(config).to(device)
    
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

    # Initialize regularizer if target-centric is enabled
    if config['target_centric']['enable']:
        logging.info("🎯 Initializing Target-Centric Prior Modeling...")
        
        target_features = data.x
        edge_index = data.edge_index if hasattr(data, 'edge_index') else None
        
        if not debug_tensor(target_features, "Target features for regularizer"):
            logging.error("❌ Target features contain NaN/Inf - disabling Target-Centric")
            config['target_centric']['enable'] = False
        else:
            try:
                loss_fn.initialize_regularizer_with_target_features(
                    target_features, encoder, edge_index
                )
                logging.info("✅ Target-Centric regularizer initialized")
            except Exception as e:
                logging.error(f"❌ Target-Centric initialization failed: {e}")
                logging.error("   Disabling Target-Centric to continue training")
                config['target_centric']['enable'] = False
        
        logging.info(f"   - Regularization: {config['target_centric']['regularization']['type']}")
        logging.info(f"   - Beta: {config['target_centric']['regularization']['beta']}")
    
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

    # Final verification before training loop
    if not debug_tensor(x, "Training features"):
        raise ValueError("Training features contain NaN/Inf")

    # Training loop
    logging.info("="*80)
    logging.info("🚀 STARTING TRAINING LOOP")
    logging.info("="*80)
    
    best_val_score = 0.0
    results = {'train_losses': [], 'val_metrics': [], 'epoch_details': []}
    
    for epoch in range(cfg['epochs']):
        # Training
        prompt.train()
        classifier.train()

        # Safe forward pass
        prompted_x, h, logits = safe_forward_pass(
            encoder, prompt, classifier, x, edge_index, f"Training epoch {epoch}"
        )
        
        if logits is None:
            logging.error(f"❌ Training failed at epoch {epoch} - forward pass returned None")
            break

        # Compute loss
        try:
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
            
        except Exception as e:
            logging.error(f"❌ Loss computation failed at epoch {epoch}: {e}")
            break
        
        # Advanced NaN detection
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error(f"❌ NaN/Inf detected in total_loss at epoch {epoch}!")
            logging.error(f"   Task Loss: {task_loss.item() if not torch.isnan(task_loss) else 'NaN'}")
            logging.error(f"   Reg Loss: {reg_loss.item() if not torch.isnan(reg_loss) else 'NaN'}")
            logging.error("   Detailed analysis:")
            debug_tensor(logits[train_mask], "Training logits")
            debug_tensor(h[train_mask], "Training embeddings")
            break

        # Backward pass
        try:
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(prompt.parameters()) + list(classifier.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
        except Exception as e:
            logging.error(f"❌ Backward pass failed at epoch {epoch}: {e}")
            break

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