"""Dynamic performance-based anchor selection for graph neural networks."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import deque

from .performance_evaluator import NodePerformanceEvaluator

logger = logging.getLogger(__name__)


class AnchorQualityTracker:
    """Tracks quality metrics for selected anchors over time."""
    
    def __init__(self, history_length: int = 10):
        """Initialize quality tracker.
        
        Args:
            history_length: Number of recent anchor selections to keep in history
        """
        self.history_length = history_length
        self.anchor_history = deque(maxlen=history_length)
        self.quality_history = deque(maxlen=history_length)
    
    def compute_anchor_quality_metrics(self, anchors: torch.Tensor, 
                                     all_embeddings: torch.Tensor,
                                     mask: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive anchor quality metrics.
        
        Args:
            anchors: Selected anchor embeddings [K, D]
            all_embeddings: All node embeddings [N, D]
            mask: Training mask [N]
            
        Returns:
            Dictionary with quality metrics
        """
        if anchors.size(0) == 0:
            return {
                'diversity': 0.0,
                'representativeness': float('inf'),
                'coverage': 0.0,
                'stability': 0.0
            }
        
        metrics = {}
        
        # 1. Diversity: How diverse are the selected anchors?
        if anchors.size(0) > 1:
            anchor_distances = torch.cdist(anchors, anchors, p=2)
            # Get upper triangle (excluding diagonal)
            upper_triangle_mask = torch.triu(torch.ones_like(anchor_distances, dtype=torch.bool), diagonal=1)
            if upper_triangle_mask.any():
                pairwise_distances = anchor_distances[upper_triangle_mask]
                metrics['diversity'] = pairwise_distances.mean().item()
            else:
                metrics['diversity'] = 0.0
        else:
            metrics['diversity'] = 0.0
        
        # 2. Representativeness: How well do anchors represent the full distribution?
        train_embeddings = all_embeddings[mask]  # [num_train, D]
        if train_embeddings.size(0) > 0:
            distances_to_anchors = torch.cdist(train_embeddings, anchors, p=2)  # [num_train, K]
            min_distances = distances_to_anchors.min(dim=1)[0]  # [num_train]
            metrics['representativeness'] = min_distances.mean().item()
            
            # 3. Coverage: Fraction of nodes that are "close" to at least one anchor
            # Define "close" as within 1 standard deviation of mean distance
            mean_dist = min_distances.mean()
            std_dist = min_distances.std()
            threshold = mean_dist + std_dist
            close_nodes = (min_distances <= threshold).float()
            metrics['coverage'] = close_nodes.mean().item()
        else:
            metrics['representativeness'] = float('inf')
            metrics['coverage'] = 0.0
        
        # 4. Stability: Consistency with previous anchor selections
        metrics['stability'] = self._compute_stability(anchors)
        
        return metrics
    
    def _compute_stability(self, current_anchors: torch.Tensor) -> float:
        """Compute stability based on overlap with previous selections.
        
        Args:
            current_anchors: Current anchor embeddings [K, D]
            
        Returns:
            Stability score (0 = completely different, 1 = identical)
        """
        if len(self.anchor_history) == 0:
            return 1.0  # First selection is perfectly stable
        
        # Compare with most recent previous selection
        prev_anchors = self.anchor_history[-1]
        
        if prev_anchors.size(0) == 0 or current_anchors.size(0) == 0:
            return 0.0
        
        # Compute pairwise distances between current and previous anchors
        distances = torch.cdist(current_anchors, prev_anchors, p=2)  # [K_curr, K_prev]
        
        # For each current anchor, find closest previous anchor
        min_distances, _ = distances.min(dim=1)  # [K_curr]
        
        # Define "matching" as distance below threshold
        # Use median distance as threshold for robustness
        all_distances = torch.cat([distances.flatten(), min_distances])
        threshold = all_distances.median() * 0.5  # Conservative threshold
        
        matching_anchors = (min_distances <= threshold).float()
        stability = matching_anchors.mean().item()
        
        return stability
    
    def update_history(self, anchors: torch.Tensor, quality_metrics: Dict[str, float]):
        """Update anchor and quality history.
        
        Args:
            anchors: Selected anchor embeddings [K, D]
            quality_metrics: Computed quality metrics
        """
        self.anchor_history.append(anchors.clone().detach())
        self.quality_history.append(quality_metrics.copy())
    
    def get_recent_quality_trends(self) -> Dict[str, List[float]]:
        """Get recent quality metric trends.
        
        Returns:
            Dictionary with lists of recent values for each metric
        """
        if len(self.quality_history) == 0:
            return {}
        
        # Extract trends for each metric
        metrics = list(self.quality_history[0].keys())
        trends = {}
        
        for metric in metrics:
            trends[metric] = [q[metric] for q in self.quality_history]
        
        return trends


class DynamicPerformanceAnchorSelector:
    """Core component for performance-based dynamic anchor selection.
    
    This class implements the main logic for selecting high-performing nodes
    as anchors during training, with adaptive quality tracking and fallback mechanisms.
    """
    
    def __init__(self, selection_ratio: float = 0.2, update_frequency: int = 10,
                 loss_weight: float = 0.7, confidence_weight: float = 0.3,
                 quality_tracking: bool = True, soft_update_momentum: float = 0.9):
        """Initialize the dynamic anchor selector.
        
        Args:
            selection_ratio: Fraction of training nodes to select as anchors (0 < ratio <= 1)
            update_frequency: Update anchors every N epochs
            loss_weight: Weight for loss-based performance scoring
            confidence_weight: Weight for confidence-based performance scoring
            quality_tracking: Enable anchor quality tracking
            soft_update_momentum: Momentum for soft anchor updates (0 = hard update, 1 = no update)
        """
        if not (0 < selection_ratio <= 1):
            raise ValueError(f"selection_ratio must be in (0, 1], got {selection_ratio}")
        if update_frequency < 1:
            raise ValueError(f"update_frequency must be >= 1, got {update_frequency}")
        if not (0 <= soft_update_momentum <= 1):
            raise ValueError(f"soft_update_momentum must be in [0, 1], got {soft_update_momentum}")
        
        self.selection_ratio = selection_ratio
        self.update_frequency = update_frequency
        self.quality_tracking = quality_tracking
        self.soft_update_momentum = soft_update_momentum
        
        # Initialize performance evaluator
        self.performance_evaluator = NodePerformanceEvaluator(
            loss_weight=loss_weight,
            confidence_weight=confidence_weight
        )
        
        # Initialize quality tracker
        if self.quality_tracking:
            self.quality_tracker = AnchorQualityTracker()
        else:
            self.quality_tracker = None
        
        # State tracking
        self.current_anchors = None
        self.last_update_epoch = -1
        self.selection_stats = []
        
        logger.info(f"DynamicPerformanceAnchorSelector initialized: "
                   f"ratio={selection_ratio}, freq={update_frequency}, "
                   f"momentum={soft_update_momentum}")
    
    def should_update_anchors(self, epoch: int) -> bool:
        """Determine if anchors should be updated at current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            True if anchors should be updated
        """
        if self.last_update_epoch == -1:  # First time
            return True
        
        return (epoch - self.last_update_epoch) >= self.update_frequency
    
    def select_anchors_by_performance(self, embeddings: torch.Tensor, logits: torch.Tensor,
                                    labels: torch.Tensor, mask: torch.Tensor, 
                                    epoch: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select anchors based on current node performance.
        
        Args:
            embeddings: Node embeddings [N, D]
            logits: Model predictions [N, C]
            labels: Ground truth labels [N]
            mask: Training mask [N]
            epoch: Current training epoch
            
        Returns:
            Tuple of:
                - Selected anchor embeddings [K, D]
                - Selection statistics and metadata
        """
        # Get indices of top-performing nodes
        top_indices = self.performance_evaluator.get_top_performing_indices(
            logits, labels, mask, self.selection_ratio
        )
        
        # Extract embeddings for selected anchors
        new_anchors = embeddings[top_indices].clone().detach()
        
        # Apply soft update if we have previous anchors
        if self.current_anchors is not None and self.soft_update_momentum > 0:
            # Ensure compatible dimensions
            if new_anchors.shape == self.current_anchors.shape:
                updated_anchors = (self.soft_update_momentum * self.current_anchors + 
                                 (1 - self.soft_update_momentum) * new_anchors)
                logger.debug(f"Applied soft update with momentum {self.soft_update_momentum}")
            else:
                # Shape mismatch, use new anchors directly
                updated_anchors = new_anchors
                logger.warning(f"Anchor shape mismatch, using hard update: "
                             f"{self.current_anchors.shape} -> {new_anchors.shape}")
        else:
            # First time or hard update
            updated_anchors = new_anchors
        
        # Compute quality metrics if enabled
        quality_metrics = {}
        if self.quality_tracking and self.quality_tracker is not None:
            quality_metrics = self.quality_tracker.compute_anchor_quality_metrics(
                updated_anchors, embeddings, mask
            )
            self.quality_tracker.update_history(updated_anchors, quality_metrics)
        
        # Compute performance statistics
        performance_stats = self.performance_evaluator.compute_performance_statistics(
            logits, labels, mask
        )
        
        # Update state
        self.current_anchors = updated_anchors
        self.last_update_epoch = epoch
        
        # Compile selection metadata
        selection_info = {
            'epoch': epoch,
            'num_anchors': updated_anchors.size(0),
            'selection_ratio': self.selection_ratio,
            'selected_indices': top_indices.cpu().tolist(),
            'anchor_shape': list(updated_anchors.shape),
            'update_type': 'soft' if self.soft_update_momentum > 0 and hasattr(self, 'current_anchors') else 'hard',
            'quality_metrics': quality_metrics,
            'performance_stats': performance_stats
        }
        
        self.selection_stats.append(selection_info)
        
        logger.info(f"✅ Selected {updated_anchors.size(0)} anchors at epoch {epoch} "
                   f"(accuracy: {performance_stats['overall_accuracy']:.3f})")
        
        if quality_metrics:
            logger.debug(f"Quality metrics - Diversity: {quality_metrics['diversity']:.3f}, "
                        f"Representativeness: {quality_metrics['representativeness']:.3f}, "
                        f"Stability: {quality_metrics['stability']:.3f}")
        
        return updated_anchors, selection_info
    
    def get_current_anchors(self) -> Optional[torch.Tensor]:
        """Get currently selected anchor embeddings.
        
        Returns:
            Current anchor embeddings [K, D] or None if not initialized
        """
        return self.current_anchors
    
    def check_anchor_quality(self, embeddings: torch.Tensor, mask: torch.Tensor,
                           performance_threshold: float = 0.05,
                           diversity_threshold: float = 0.3,
                           stability_threshold: float = 0.5) -> Dict[str, bool]:
        """Check if current anchors meet quality criteria.
        
        Args:
            embeddings: Current node embeddings [N, D]
            mask: Training mask [N]
            performance_threshold: Minimum required performance improvement
            diversity_threshold: Minimum required anchor diversity
            stability_threshold: Minimum required selection stability
            
        Returns:
            Dictionary with quality check results
        """
        if self.current_anchors is None:
            return {
                'has_anchors': False,
                'performance_ok': False,
                'diversity_ok': False,
                'stability_ok': False,
                'overall_ok': False
            }
        
        checks = {'has_anchors': True}
        
        if self.quality_tracking and self.quality_tracker is not None:
            quality_metrics = self.quality_tracker.compute_anchor_quality_metrics(
                self.current_anchors, embeddings, mask
            )
            
            checks['diversity_ok'] = quality_metrics['diversity'] >= diversity_threshold
            checks['stability_ok'] = quality_metrics['stability'] >= stability_threshold
            
            # Check performance trend
            trends = self.quality_tracker.get_recent_quality_trends()
            if len(trends.get('representativeness', [])) >= 2:
                recent_repr = trends['representativeness'][-2:]
                performance_improved = recent_repr[-1] < recent_repr[-2]  # Lower is better
                checks['performance_ok'] = performance_improved
            else:
                checks['performance_ok'] = True  # No trend available, assume OK
        else:
            # No quality tracking, assume all checks pass
            checks['diversity_ok'] = True
            checks['stability_ok'] = True
            checks['performance_ok'] = True
        
        checks['overall_ok'] = all([
            checks['performance_ok'],
            checks['diversity_ok'],
            checks['stability_ok']
        ])
        
        return checks
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive selection statistics.
        
        Returns:
            Dictionary with selection history and trends
        """
        if not self.selection_stats:
            return {'num_selections': 0}
        
        stats = {
            'num_selections': len(self.selection_stats),
            'latest_selection': self.selection_stats[-1],
            'selection_frequency': self.update_frequency,
            'current_anchors_shape': list(self.current_anchors.shape) if self.current_anchors is not None else None
        }
        
        # Add quality trends if available
        if self.quality_tracking and self.quality_tracker is not None:
            stats['quality_trends'] = self.quality_tracker.get_recent_quality_trends()
        
        # Compute average metrics across selections
        if len(self.selection_stats) > 1:
            recent_stats = self.selection_stats[-5:]  # Last 5 selections
            
            # Average performance metrics
            accuracies = [s['performance_stats']['overall_accuracy'] for s in recent_stats]
            stats['recent_avg_accuracy'] = sum(accuracies) / len(accuracies)
            
            # Quality metrics trends (if available)
            if all('quality_metrics' in s for s in recent_stats):
                for metric in ['diversity', 'representativeness', 'stability']:
                    values = [s['quality_metrics'][metric] for s in recent_stats]
                    stats[f'recent_avg_{metric}'] = sum(values) / len(values)
        
        return stats
    
    def reset(self):
        """Reset the anchor selector state."""
        self.current_anchors = None
        self.last_update_epoch = -1
        self.selection_stats.clear()
        
        if self.quality_tracker is not None:
            self.quality_tracker.anchor_history.clear()
            self.quality_tracker.quality_history.clear()
        
        logger.info("DynamicPerformanceAnchorSelector state reset")