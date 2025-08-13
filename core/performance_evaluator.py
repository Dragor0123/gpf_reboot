"""Node performance evaluation for dynamic anchor selection."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NodePerformanceEvaluator:
    """Evaluates individual node performance for anchor selection.
    
    This class provides methods to compute various performance metrics
    for nodes, which are used to identify high-performing nodes that
    can serve as effective anchors for regularization.
    """
    
    def __init__(self, loss_weight: float = 0.7, confidence_weight: float = 0.3):
        """Initialize the performance evaluator.
        
        Args:
            loss_weight: Weight for loss-based scoring (higher = more importance)
            confidence_weight: Weight for confidence-based scoring
        """
        if abs(loss_weight + confidence_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {loss_weight + confidence_weight}")
        
        self.loss_weight = loss_weight
        self.confidence_weight = confidence_weight
        
        logger.info(f"NodePerformanceEvaluator initialized with loss_weight={loss_weight}, "
                   f"confidence_weight={confidence_weight}")
    
    def compute_loss_scores(self, logits: torch.Tensor, labels: torch.Tensor, 
                           mask: torch.Tensor) -> torch.Tensor:
        """Compute performance scores based on individual node loss.
        
        Nodes with lower individual loss demonstrate better task alignment
        and are considered higher-performing anchors.
        
        Args:
            logits: Model predictions [N, C]
            labels: Ground truth labels [N]
            mask: Training mask to select nodes [N]
            
        Returns:
            Loss-based performance scores [num_masked_nodes]
            Higher scores indicate better performance (lower loss)
        """
        # Compute individual node losses
        node_losses = F.cross_entropy(logits, labels, reduction='none')  # [N]
        train_losses = node_losses[mask]  # [num_masked_nodes]
        
        if train_losses.numel() == 0:
            raise ValueError("No training nodes available for loss computation")
        
        # Invert and normalize: lower loss → higher score
        min_loss, max_loss = train_losses.min(), train_losses.max()
        
        if torch.isclose(min_loss, max_loss):
            # All losses are the same, return uniform scores
            scores = torch.ones_like(train_losses)
            logger.warning("All node losses are identical, returning uniform scores")
        else:
            normalized = (train_losses - min_loss) / (max_loss - min_loss + 1e-8)
            scores = 1.0 - normalized  # Invert: lower loss = higher score
        
        return scores
    
    def compute_confidence_scores(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute performance scores based on prediction confidence.
        
        High confidence predictions indicate stable, reliable representations
        that make good anchor candidates.
        
        Args:
            logits: Model predictions [N, C]
            mask: Training mask to select nodes [N]
            
        Returns:
            Confidence-based performance scores [num_masked_nodes]
            Higher scores indicate higher confidence (lower entropy)
        """
        # Compute prediction probabilities
        probs = F.softmax(logits, dim=1)  # [N, C]
        
        # Compute entropy (lower entropy = higher confidence)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [N]
        train_entropy = entropy[mask]  # [num_masked_nodes]
        
        if train_entropy.numel() == 0:
            raise ValueError("No training nodes available for confidence computation")
        
        # Convert entropy to confidence score
        confidence = -train_entropy  # Lower entropy = higher confidence
        
        # Normalize to [0, 1] range
        min_conf, max_conf = confidence.min(), confidence.max()
        
        if torch.isclose(min_conf, max_conf):
            # All confidences are the same, return uniform scores
            scores = torch.ones_like(confidence)
            logger.warning("All node confidences are identical, returning uniform scores")
        else:
            scores = (confidence - min_conf) / (max_conf - min_conf + 1e-8)
        
        return scores
    
    def compute_combined_scores(self, logits: torch.Tensor, labels: torch.Tensor, 
                               mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute combined performance scores using both loss and confidence.
        
        Args:
            logits: Model predictions [N, C]
            labels: Ground truth labels [N]
            mask: Training mask to select nodes [N]
            
        Returns:
            Tuple of:
                - Combined performance scores [num_masked_nodes]
                - Dictionary with individual score components for debugging
        """
        # Compute individual score components
        loss_scores = self.compute_loss_scores(logits, labels, mask)
        confidence_scores = self.compute_confidence_scores(logits, mask)
        
        # Weighted combination
        combined_scores = (self.loss_weight * loss_scores + 
                          self.confidence_weight * confidence_scores)
        
        # Debug information
        score_info = {
            'loss_scores': loss_scores,
            'confidence_scores': confidence_scores,
            'combined_scores': combined_scores,
            'loss_weight': self.loss_weight,
            'confidence_weight': self.confidence_weight,
            'num_nodes': mask.sum().item()
        }
        
        return combined_scores, score_info
    
    def evaluate_node_performance(self, logits: torch.Tensor, labels: torch.Tensor, 
                                 mask: torch.Tensor, 
                                 return_details: bool = False) -> torch.Tensor:
        """Main interface for evaluating node performance.
        
        Args:
            logits: Model predictions [N, C]
            labels: Ground truth labels [N]
            mask: Training mask to select nodes [N]
            return_details: If True, also return detailed score breakdown
            
        Returns:
            Performance scores [num_masked_nodes]
            If return_details=True, returns (scores, details_dict)
        """
        if return_details:
            return self.compute_combined_scores(logits, labels, mask)
        else:
            scores, _ = self.compute_combined_scores(logits, labels, mask)
            return scores
    
    def get_top_performing_indices(self, logits: torch.Tensor, labels: torch.Tensor,
                                  mask: torch.Tensor, selection_ratio: float) -> torch.Tensor:
        """Get indices of top-performing nodes based on combined scores.
        
        Args:
            logits: Model predictions [N, C]
            labels: Ground truth labels [N]
            mask: Training mask to select nodes [N]
            selection_ratio: Fraction of nodes to select (0 < ratio <= 1)
            
        Returns:
            Indices of top-performing nodes in the full graph [num_selected]
        """
        if not (0 < selection_ratio <= 1):
            raise ValueError(f"selection_ratio must be in (0, 1], got {selection_ratio}")
        
        # Get performance scores
        scores = self.evaluate_node_performance(logits, labels, mask)
        
        # Determine number of anchors to select
        num_train_nodes = mask.sum().item()
        num_selected = max(1, int(num_train_nodes * selection_ratio))
        num_selected = min(num_selected, num_train_nodes)  # Cap at available nodes
        
        # Get top-k scores
        top_scores, top_local_indices = torch.topk(scores, num_selected, largest=True)
        
        # Convert local indices (within masked nodes) to global indices
        masked_node_indices = torch.where(mask)[0]  # Global indices of masked nodes
        top_global_indices = masked_node_indices[top_local_indices]
        
        logger.debug(f"Selected {num_selected}/{num_train_nodes} top-performing nodes "
                    f"(ratio={selection_ratio:.3f})")
        
        return top_global_indices
    
    def compute_performance_statistics(self, logits: torch.Tensor, labels: torch.Tensor,
                                     mask: torch.Tensor) -> dict:
        """Compute detailed performance statistics for analysis.
        
        Args:
            logits: Model predictions [N, C]
            labels: Ground truth labels [N]
            mask: Training mask to select nodes [N]
            
        Returns:
            Dictionary with performance statistics
        """
        scores, score_info = self.compute_combined_scores(logits, labels, mask)
        
        # Compute accuracy for masked nodes
        masked_logits = logits[mask]
        masked_labels = labels[mask]
        predictions = masked_logits.argmax(dim=1)
        accuracy = (predictions == masked_labels).float().mean().item()
        
        # Individual loss statistics
        node_losses = F.cross_entropy(logits, labels, reduction='none')
        masked_losses = node_losses[mask]
        
        statistics = {
            'overall_accuracy': accuracy,
            'num_nodes': mask.sum().item(),
            'performance_scores': {
                'mean': scores.mean().item(),
                'std': scores.std().item(),
                'min': scores.min().item(),
                'max': scores.max().item(),
            },
            'loss_scores': {
                'mean': score_info['loss_scores'].mean().item(),
                'std': score_info['loss_scores'].std().item(),
            },
            'confidence_scores': {
                'mean': score_info['confidence_scores'].mean().item(),
                'std': score_info['confidence_scores'].std().item(),
            },
            'raw_losses': {
                'mean': masked_losses.mean().item(),
                'std': masked_losses.std().item(),
                'min': masked_losses.min().item(),
                'max': masked_losses.max().item(),
            }
        }
        
        return statistics