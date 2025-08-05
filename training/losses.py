"""Loss functions for GPF training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute cross-entropy loss with optional masking.
    
    Args:
        logits: Model predictions [N, C]
        labels: Ground truth labels [N]
        mask: Optional mask for selecting nodes [N]
        
    Returns:
        Cross-entropy loss (scalar)
    """
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    
    return F.cross_entropy(logits, labels)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel between two batches of vectors.
    
    Args:
        x: First batch [N, D]
        y: Second batch [M, D]
        sigma: Kernel bandwidth
        
    Returns:
        Kernel matrix [N, M]
    """
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-dist / (2 * sigma ** 2 + 1e-8))


def compute_mmd_loss(x_samples: torch.Tensor, y_samples: torch.Tensor, 
                    sigma: float = 1.0) -> torch.Tensor:
    """Compute Maximum Mean Discrepancy (MMD) between two distributions.
    
    Args:
        x_samples: Samples from first distribution [N, D]
        y_samples: Samples from second distribution [M, D]
        sigma: Kernel bandwidth
        
    Returns:
        MMD loss (scalar)
    """
    K_xx = gaussian_kernel(x_samples, x_samples, sigma)
    K_yy = gaussian_kernel(y_samples, y_samples, sigma)
    K_xy = gaussian_kernel(x_samples, y_samples, sigma)
    
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def compute_unbiased_mmd_loss(x_samples: torch.Tensor, y_samples: torch.Tensor, 
                             sigma: float = 1.0) -> torch.Tensor:
    """Compute unbiased MMD estimate.
    
    Args:
        x_samples: Samples from first distribution [N, D]
        y_samples: Samples from second distribution [M, D]
        sigma: Kernel bandwidth
        
    Returns:
        Unbiased MMD loss (scalar)
    """
    K_xx = gaussian_kernel(x_samples, x_samples, sigma)
    K_yy = gaussian_kernel(y_samples, y_samples, sigma)
    K_xy = gaussian_kernel(x_samples, y_samples, sigma)
    
    m = x_samples.size(0)
    n = y_samples.size(0)
    
    # Unbiased estimate
    mmd = 0.0
    if m > 1:
        mmd += (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1))
    if n > 1:
        mmd += (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1))
    mmd -= 2 * K_xy.mean()
    
    return mmd


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for pretraining."""
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss between two views.
        
        Args:
            z1: First view embeddings [N, D]
            z2: Second view embeddings [N, D]
            
        Returns:
            InfoNCE loss (scalar)
        """
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Positive pairs are on the diagonal
        positives = torch.diag(sim_matrix)
        
        # InfoNCE loss
        numerator = torch.exp(positives)
        denominator = torch.sum(torch.exp(sim_matrix), dim=1)
        loss = -torch.log(numerator / denominator)
        
        return loss.mean()


class TargetCentricLoss(nn.Module):
    """Enhanced Target-Centric Prior Modeling Loss Function."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.target_centric_enabled = config.get('target_centric_enable', False)
        
        if self.target_centric_enabled:
            # Import here to avoid circular imports
            from .regularizers import TargetCentricRegularizer
            
            reg_config = {
                'beta': config.get('target_centric_beta', 0.1),
                'anchor': {
                    'type': config.get('target_centric_anchor_type', 'random'),
                    'num_anchors': config.get('target_centric_anchor_num_anchors', 500),
                    'num_components': config.get('target_centric_anchor_num_components', 8),
                    'use_sklearn_gmm': config.get('target_centric_anchor_use_sklearn_gmm', True),
                },
                'mapper': {
                    'type': config.get('target_centric_mapper_type', 'identity'),
                },
                'divergence': {
                    'type': config.get('target_centric_divergence_type', 'mmd'),
                    'params': {
                        'sigma': config.get('target_centric_divergence_sigma', 0.25),
                    }
                }
            }
            
            self.regularizer = TargetCentricRegularizer(reg_config)
        else:
            self.regularizer = None
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                embeddings: torch.Tensor, mask: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute total loss.
        
        Args:
            logits: Model predictions [N, C]
            labels: True labels [N]
            embeddings: Node embeddings [N, D]
            mask: Training mask [N]
            edge_index: Graph edges [2, E] (optional)
            
        Returns:
            Dictionary with loss components
        """
        # Task alignment loss
        task_loss = compute_ce_loss(logits, labels, mask)
        
        losses = {
            'task_loss': task_loss,
            'total_loss': task_loss
        }
        
        # Add regularization if enabled
        if self.target_centric_enabled and self.regularizer is not None:
            reg_loss = self.regularizer(embeddings)
            losses['reg_loss'] = reg_loss
            losses['total_loss'] = task_loss + reg_loss
        else:
            losses['reg_loss'] = torch.tensor(0.0, device=task_loss.device)
        
        return losses
    
    def initialize_regularizer_with_target_features(self, target_features: torch.Tensor, 
                                                   encoder: nn.Module, 
                                                   edge_index: Optional[torch.Tensor] = None):
        """Initialize regularizer with target features.
        
        Args:
            target_features: Original target features [N, D]
            encoder: Graph encoder model
            edge_index: Graph edges [2, E] (optional)
        """
        if self.regularizer is not None:
            logger.info("Initializing Target-Centric regularizer with target features")
            self.regularizer.initialize_anchors(target_features, encoder, edge_index)
    
    def initialize_regularizer_with_fixed_anchors(self, anchors: torch.Tensor):
        """Initialize regularizer with pre-generated anchors.
        
        Args:
            anchors: Fixed anchor vectors [K, D]
        """
        if self.regularizer is not None:
            logger.info(f"Initializing regularizer with {anchors.size(0)} fixed anchors")
            self.regularizer.initialize_fixed_anchors(anchors)