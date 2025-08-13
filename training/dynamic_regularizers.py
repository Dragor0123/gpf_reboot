"""Dynamic anchor-based regularization for graph neural networks."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dynamic_anchor_selector import DynamicPerformanceAnchorSelector
from .regularizers import DivergenceMetric, MMDDivergence, WassersteinDivergence, create_divergence_metric

logger = logging.getLogger(__name__)


class DynamicAnchorRegularizer(nn.Module):
    """Dynamic anchor-based regularizer with fallback mechanisms.
    
    This regularizer uses performance-based dynamic anchor selection to create
    adaptive reference distributions for cross-domain graph learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dynamic anchor regularizer.
        
        Args:
            config: Configuration dictionary containing:
                - selection_ratio: Fraction of nodes to select as anchors
                - update_frequency: Epochs between anchor updates
                - criteria: loss_weight and confidence_weight for scoring
                - quality_tracking: Enable quality metrics tracking
                - fallback: Fallback configuration
                - divergence: Divergence metric configuration
                - beta: Regularization strength
        """
        super().__init__()
        
        # Extract configuration
        self.config = config
        self.beta = config.get('beta', 0.1)
        
        # Dynamic anchor selector configuration
        selector_config = config.get('dynamic_anchor', {})
        criteria_config = selector_config.get('criteria', {})
        
        self.anchor_selector = DynamicPerformanceAnchorSelector(
            selection_ratio=selector_config.get('selection_ratio', 0.2),
            update_frequency=selector_config.get('update_frequency', 10),
            loss_weight=criteria_config.get('loss_weight', 0.7),
            confidence_weight=criteria_config.get('confidence_weight', 0.3),
            quality_tracking=selector_config.get('quality_tracking', True),
            soft_update_momentum=selector_config.get('soft_update_momentum', 0.9)
        )
        
        # Divergence metric for computing regularization loss
        divergence_config = config.get('divergence', {})
        self.divergence_metric = create_divergence_metric(
            divergence_config.get('type', 'mmd'),
            **divergence_config.get('params', {})
        )
        
        # Fallback configuration
        self.fallback_config = config.get('fallback', {})
        self.fallback_enabled = self.fallback_config.get('enable', True)
        self.fallback_method = self.fallback_config.get('method', 'mog')
        self.fallback_conditions = self.fallback_config.get('conditions', {})
        
        # Fallback anchors (from MoG or other static methods)
        self.fallback_anchors = None
        self.is_using_fallback = False
        self.fallback_reason = None
        
        # State tracking
        self.initialized = False
        self.last_performance = None
        
        logger.info(f"DynamicAnchorRegularizer initialized with beta={self.beta}")
        logger.info(f"Fallback enabled: {self.fallback_enabled} (method: {self.fallback_method})")
    
    def set_fallback_anchors(self, anchors: torch.Tensor):
        """Set fallback anchors for emergency use.
        
        Args:
            anchors: Pre-generated anchor vectors [K, D]
        """
        self.fallback_anchors = anchors.detach()
        logger.info(f"Fallback anchors set with shape: {self.fallback_anchors.shape}")
    
    def _check_fallback_conditions(self, embeddings: torch.Tensor, mask: torch.Tensor,
                                  current_performance: float) -> Tuple[bool, Optional[str]]:
        """Check if fallback should be triggered.
        
        Args:
            embeddings: Current node embeddings [N, D]
            mask: Training mask [N]
            current_performance: Current model performance (accuracy)
            
        Returns:
            Tuple of (should_fallback, reason)
        """
        if not self.fallback_enabled:
            return False, None
        
        conditions = self.fallback_conditions
        
        # Check performance drop
        if self.last_performance is not None:
            performance_drop_threshold = conditions.get('performance_drop', 0.05)
            performance_drop = self.last_performance - current_performance
            
            if performance_drop > performance_drop_threshold:
                return True, f"Performance dropped by {performance_drop:.3f} > {performance_drop_threshold}"
        
        # Check anchor quality
        quality_checks = self.anchor_selector.check_anchor_quality(
            embeddings, mask,
            diversity_threshold=conditions.get('anchor_diversity', 0.3),
            stability_threshold=conditions.get('selection_instability', 0.5)
        )
        
        if not quality_checks['diversity_ok']:
            return True, f"Anchor diversity too low: {quality_checks.get('diversity', 'N/A')}"
        
        if not quality_checks['stability_ok']:
            return True, f"Anchor selection too unstable: {quality_checks.get('stability', 'N/A')}"
        
        return False, None
    
    def _activate_fallback(self, reason: str):
        """Activate fallback mode.
        
        Args:
            reason: Reason for activating fallback
        """
        if self.fallback_anchors is None:
            logger.error(f"❌ Fallback triggered ({reason}) but no fallback anchors available!")
            return
        
        self.is_using_fallback = True
        self.fallback_reason = reason
        
        logger.warning(f"🔄 Fallback activated: {reason}")
        logger.info(f"Switching to {self.fallback_method} anchors with shape: {self.fallback_anchors.shape}")
    
    def _deactivate_fallback(self):
        """Deactivate fallback mode."""
        if self.is_using_fallback:
            self.is_using_fallback = False
            self.fallback_reason = None
            logger.info("✅ Fallback deactivated, returning to dynamic anchor selection")
    
    def update_anchors(self, embeddings: torch.Tensor, logits: torch.Tensor,
                      labels: torch.Tensor, mask: torch.Tensor, epoch: int) -> Dict[str, Any]:
        """Update anchors using dynamic selection or fallback.
        
        Args:
            embeddings: Node embeddings [N, D]
            logits: Model predictions [N, C]
            labels: Ground truth labels [N]
            mask: Training mask [N]
            epoch: Current training epoch
            
        Returns:
            Dictionary with update information
        """
        # Compute current performance
        with torch.no_grad():
            masked_logits = logits[mask]
            masked_labels = labels[mask]
            predictions = masked_logits.argmax(dim=1)
            current_performance = (predictions == masked_labels).float().mean().item()
        
        update_info = {
            'epoch': epoch,
            'current_performance': current_performance,
            'anchors_updated': False,
            'using_fallback': self.is_using_fallback,
            'fallback_reason': self.fallback_reason
        }
        
        # Check if we should update anchors
        should_update = self.anchor_selector.should_update_anchors(epoch)
        
        if should_update:
            # Check fallback conditions
            should_fallback, fallback_reason = self._check_fallback_conditions(
                embeddings, mask, current_performance
            )
            
            if should_fallback and not self.is_using_fallback:
                self._activate_fallback(fallback_reason)
                update_info['fallback_activated'] = True
                update_info['fallback_reason'] = fallback_reason
            elif not should_fallback and self.is_using_fallback:
                self._deactivate_fallback()
                update_info['fallback_deactivated'] = True
            
            # Update anchors (either dynamic or keep using fallback)
            if not self.is_using_fallback:
                try:
                    new_anchors, selection_info = self.anchor_selector.select_anchors_by_performance(
                        embeddings, logits, labels, mask, epoch
                    )
                    update_info['anchors_updated'] = True
                    update_info['selection_info'] = selection_info
                    update_info['num_anchors'] = new_anchors.size(0)
                    
                except Exception as e:
                    logger.error(f"❌ Dynamic anchor selection failed: {e}")
                    if self.fallback_enabled and self.fallback_anchors is not None:
                        self._activate_fallback(f"Selection error: {str(e)}")
                        update_info['fallback_activated'] = True
                        update_info['fallback_reason'] = f"Selection error: {str(e)}"
                    else:
                        raise e
        
        # Update performance history
        self.last_performance = current_performance
        
        if not self.initialized:
            self.initialized = True
            logger.info("✅ DynamicAnchorRegularizer initialization completed")
        
        return update_info
    
    def get_current_anchors(self) -> torch.Tensor:
        """Get current anchor embeddings.
        
        Returns:
            Current anchor embeddings [K, D]
        """
        if self.is_using_fallback:
            if self.fallback_anchors is None:
                raise RuntimeError("No fallback anchors available")
            return self.fallback_anchors
        else:
            current_anchors = self.anchor_selector.get_current_anchors()
            if current_anchors is None:
                if self.fallback_anchors is not None:
                    logger.warning("No dynamic anchors available, using fallback")
                    return self.fallback_anchors
                else:
                    raise RuntimeError("No anchors available (neither dynamic nor fallback)")
            return current_anchors
    
    def compute_regularization_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss using current anchors.
        
        Args:
            embeddings: Node embeddings [N, D]
            
        Returns:
            Regularization loss (scalar)
        """
        current_anchors = self.get_current_anchors()
        
        # Compute divergence between embeddings and anchors
        divergence = self.divergence_metric.compute_divergence(embeddings, current_anchors)
        
        return self.beta * divergence
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute regularization loss.
        
        Args:
            embeddings: Node embeddings [N, D]
            
        Returns:
            Regularization loss (scalar)
        """
        return self.compute_regularization_loss(embeddings)
    
    def get_regularizer_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the regularizer.
        
        Returns:
            Dictionary with regularizer status and statistics
        """
        status = {
            'initialized': self.initialized,
            'using_fallback': self.is_using_fallback,
            'fallback_reason': self.fallback_reason,
            'fallback_enabled': self.fallback_enabled,
            'fallback_method': self.fallback_method,
            'beta': self.beta,
            'divergence_type': type(self.divergence_metric).__name__
        }
        
        # Add current anchor information
        try:
            current_anchors = self.get_current_anchors()
            status['current_anchors_shape'] = list(current_anchors.shape)
            status['has_anchors'] = True
        except RuntimeError:
            status['current_anchors_shape'] = None
            status['has_anchors'] = False
        
        # Add selection statistics if available
        if not self.is_using_fallback:
            status['selection_stats'] = self.anchor_selector.get_selection_statistics()
        
        return status
    
    def reset(self):
        """Reset the regularizer state."""
        self.anchor_selector.reset()
        self.is_using_fallback = False
        self.fallback_reason = None
        self.initialized = False
        self.last_performance = None
        
        logger.info("DynamicAnchorRegularizer state reset")


def create_dynamic_anchor_regularizer(config: Dict[str, Any]) -> DynamicAnchorRegularizer:
    """Factory function for creating dynamic anchor regularizer.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Configured DynamicAnchorRegularizer instance
    """
    # Extract relevant configuration sections
    dynamic_config = {
        'dynamic_anchor': config.get('dynamic_anchor', {}),
        'target_centric': config.get('target_centric', {}),
        'fallback': config.get('dynamic_anchor', {}).get('fallback', {}),
        'divergence': config.get('target_centric', {}).get('regularization', {}).get('divergence', {}),
        'beta': config.get('target_centric', {}).get('regularization', {}).get('weight', 0.1)
    }
    
    return DynamicAnchorRegularizer(dynamic_config)