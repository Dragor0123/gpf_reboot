"""Target-centric regularization components for GPF training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AnchorSelector(ABC):
    """Abstract base class for anchor selection strategies."""
    
    @abstractmethod
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        pass


class AnchorMapper(ABC):
    """Abstract base class for mapping anchors to hidden space."""
    
    @abstractmethod
    def initialize(self, anchor_features: torch.Tensor, 
                  encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        pass
    
    @abstractmethod
    def get_anchor_representations(self) -> torch.Tensor:
        pass


class DivergenceMetric(ABC):
    """Abstract base class for divergence metrics."""
    
    @abstractmethod
    def compute_divergence(self, prompted_embeddings: torch.Tensor,
                          anchor_representations: torch.Tensor) -> torch.Tensor:
        pass


class RandomAnchorSelector(AnchorSelector):
    """Random sampling of anchor nodes."""
    
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        num_nodes = target_features.size(0)
        num_anchors = min(num_anchors, num_nodes)
        indices = torch.randperm(num_nodes)[:num_anchors]
        return target_features[indices].clone()


class HighDegreeAnchorSelector(AnchorSelector):
    """Select high-degree nodes as anchors."""
    
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        if edge_index is None:
            logger.warning("No edge_index provided, falling back to random selection")
            return RandomAnchorSelector().select_anchors(target_features, None, num_anchors)
        
        # Compute node degrees
        num_nodes = target_features.size(0)
        degrees = torch.zeros(num_nodes, device=target_features.device)
        degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        
        # Select top-degree nodes
        num_anchors = min(num_anchors, num_nodes)
        _, top_indices = torch.topk(degrees, num_anchors)
        return target_features[top_indices].clone()


class GradientOptimizedAnchorSelector(AnchorSelector):
    """Gradient-based optimization for anchor generation."""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 100, 
                 regularization_lambda: float = 0.1, objective_weights: Optional[Dict[str, float]] = None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_lambda = regularization_lambda
        self.objective_weights = objective_weights or {
            'high_norm': 0.3,
            'diversity': 0.4, 
            'task_alignment': 0.3
        }
    
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        """Generate anchors using gradient-based optimization."""
        device = target_features.device
        feature_dim = target_features.size(1)
        num_anchors = min(num_anchors, target_features.size(0))
        
        # Initialize with random subset of target features
        initial_indices = torch.randperm(target_features.size(0))[:num_anchors]
        initial_features = target_features[initial_indices].clone()
        
        # Create optimizable features
        optimized_features = initial_features.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_features], lr=self.learning_rate)
        
        # Optimization loop
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Compute objectives
            loss = self._compute_multi_objective(optimized_features, initial_features, edge_index)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log progress
            if iteration % 20 == 0:
                logger.debug(f"Gradient optimization iteration {iteration}: loss = {loss.item():.4f}")
        
        return optimized_features.detach()
    
    def _compute_multi_objective(self, features: torch.Tensor, original_features: torch.Tensor,
                                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-objective loss for optimization."""
        total_loss = 0.0
        
        # Objective 1: High norm (encourage strong activations)
        if 'high_norm' in self.objective_weights:
            high_norm_objective = -torch.norm(features, dim=1).mean()  # Negative for maximization
            total_loss += self.objective_weights['high_norm'] * high_norm_objective
        
        # Objective 2: Feature diversity (maximize pairwise distances)
        if 'diversity' in self.objective_weights:
            diversity_objective = -self._compute_diversity(features)  # Negative for maximization
            total_loss += self.objective_weights['diversity'] * diversity_objective
        
        # Objective 3: Task alignment (preserve target distribution characteristics)
        if 'task_alignment' in self.objective_weights:
            alignment_objective = self._compute_task_alignment(features, original_features)
            total_loss += self.objective_weights['task_alignment'] * alignment_objective
        
        # Regularization: prevent features from deviating too far from originals
        regularization = torch.norm(features - original_features, p=2)
        total_loss += self.regularization_lambda * regularization
        
        return total_loss
    
    def _compute_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute diversity as mean pairwise distance."""
        pairwise_distances = torch.cdist(features, features, p=2)
        # Exclude diagonal (self-distances)
        mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        return pairwise_distances[mask].mean()
    
    def _compute_task_alignment(self, features: torch.Tensor, original_features: torch.Tensor) -> torch.Tensor:
        """Compute task alignment by preserving statistical properties."""
        # Mean preservation
        mean_diff = torch.norm(features.mean(dim=0) - original_features.mean(dim=0), p=2)
        
        # Variance preservation
        var_diff = torch.norm(features.var(dim=0) - original_features.var(dim=0), p=2)
        
        return mean_diff + var_diff


class IdentityAnchorMapper(AnchorMapper):
    """Identity mapping for pre-generated anchors."""
    
    def __init__(self):
        self.anchor_representations = None
    
    def initialize(self, anchor_features: torch.Tensor, 
                  encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        """Store anchor features directly."""
        self.anchor_representations = anchor_features.clone()
    
    def get_anchor_representations(self) -> torch.Tensor:
        if self.anchor_representations is None:
            raise RuntimeError("Anchor mapper not initialized")
        return self.anchor_representations


class EncoderAnchorMapper(AnchorMapper):
    """Map anchors using the frozen encoder."""
    
    def __init__(self):
        self.anchor_representations = None
    
    def initialize(self, anchor_features: torch.Tensor, 
                  encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        """Initialize anchor representations using encoder."""
        with torch.no_grad():
            encoder.eval()
            
            # Create empty edge index for isolated anchor encoding
            device = anchor_features.device
            empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            anchor_embeddings = encoder(anchor_features, empty_edge_index)
            
            self.anchor_representations = anchor_embeddings.clone()
    
    def get_anchor_representations(self) -> torch.Tensor:
        if self.anchor_representations is None:
            raise RuntimeError("Anchor mapper not initialized")
        return self.anchor_representations


class MMDDivergence(DivergenceMetric):
    """MMD divergence metric."""
    
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
    
    def compute_divergence(self, prompted_embeddings: torch.Tensor,
                          anchor_representations: torch.Tensor) -> torch.Tensor:
        return self._compute_unbiased_mmd_loss(prompted_embeddings, anchor_representations, self.sigma)
    
    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Compute RBF (Gaussian) kernel between two batches of vectors."""
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * sigma ** 2 + 1e-8))
    
    def _compute_unbiased_mmd_loss(self, x_samples: torch.Tensor, y_samples: torch.Tensor, 
                                   sigma: float = 1.0) -> torch.Tensor:
        """Compute unbiased MMD estimate."""
        K_xx = self._gaussian_kernel(x_samples, x_samples, sigma)
        K_yy = self._gaussian_kernel(y_samples, y_samples, sigma)
        K_xy = self._gaussian_kernel(x_samples, y_samples, sigma)
        
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


class WassersteinDivergence(DivergenceMetric):
    """Wasserstein divergence metric."""
    
    def compute_divergence(self, prompted_embeddings: torch.Tensor,
                          anchor_representations: torch.Tensor) -> torch.Tensor:
        return torch.norm(prompted_embeddings.mean(dim=0) - anchor_representations.mean(dim=0), p=2)


def create_anchor_selector(selector_type: str, **kwargs) -> AnchorSelector:
    """Factory function for anchor selectors."""
    selectors = {
        'random': RandomAnchorSelector,
        'high_degree': HighDegreeAnchorSelector,
        'gradient_optimized': GradientOptimizedAnchorSelector,
    }
    
    if selector_type not in selectors:
        raise ValueError(f"Unknown anchor selector: {selector_type}")
    
    if selector_type == 'gradient_optimized':
        return selectors[selector_type](**kwargs)
    else:
        return selectors[selector_type]()


def create_anchor_mapper(mapper_type: str, **kwargs) -> AnchorMapper:
    """Factory function for anchor mappers."""
    mappers = {
        'identity': IdentityAnchorMapper,
        'encoder': EncoderAnchorMapper,
    }
    
    if mapper_type not in mappers:
        raise ValueError(f"Unknown anchor mapper: {mapper_type}")
    
    return mappers[mapper_type]()


def create_divergence_metric(metric_type: str, **kwargs) -> DivergenceMetric:
    """Factory function for divergence metrics."""
    metrics = {
        'mmd': lambda: MMDDivergence(kwargs.get('sigma', 1.0)),
        'wasserstein': WassersteinDivergence,
    }
    
    if metric_type not in metrics:
        raise ValueError(f"Unknown divergence metric: {metric_type}")
    
    return metrics[metric_type]()


class TargetCentricRegularizer(nn.Module):
    """Target-Centric Prior Modeling Regularizer with flexible design."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Parse configuration with defaults
        anchor_config = config.get('anchor', {})
        mapper_config = config.get('mapper', {})
        divergence_config = config.get('divergence', {})
        
        # Handle fixed anchor mode (gaussian/mog)
        anchor_type = anchor_config.get('type', 'random')
        self.is_fixed_anchor_mode = (anchor_type in ['gaussian', 'mog'])
        
        if self.is_fixed_anchor_mode:
            self.anchor_selector = None
            self.anchor_mapper = create_anchor_mapper('identity')
        else:
            # Pass optimization parameters for gradient_optimized anchor selector
            optimization_config = anchor_config.get('optimization', {})
            anchor_kwargs = {}
            if anchor_type == 'gradient_optimized':
                anchor_kwargs = {
                    'learning_rate': optimization_config.get('learning_rate', 0.01),
                    'num_iterations': optimization_config.get('num_iterations', 100),
                    'regularization_lambda': optimization_config.get('regularization_lambda', 0.1),
                    'objective_weights': optimization_config.get('objectives', {
                        'high_norm': 0.3,
                        'diversity': 0.4,
                        'task_alignment': 0.3
                    })
                }
            
            self.anchor_selector = create_anchor_selector(anchor_type, **anchor_kwargs)
            self.anchor_mapper = create_anchor_mapper(
                mapper_config.get('type', 'encoder')
            )

        self.divergence_metric = create_divergence_metric(
            divergence_config.get('type', 'mmd'),
            **divergence_config.get('params', {})
        )
        
        self.beta = config.get('beta', 0.1)
        self.num_anchors = anchor_config.get('num_anchors', 100)
        self.fixed_anchors = None

    def initialize_fixed_anchors(self, anchor_vectors: torch.Tensor):
        """Initialize with pre-generated anchor vectors."""
        self.fixed_anchors = anchor_vectors.detach()
        self.anchor_mapper.initialize(self.fixed_anchors, None, None)
        logger.info(f"✅ Fixed anchors registered with shape: {self.fixed_anchors.shape}")

    def initialize_anchors(self, target_features: torch.Tensor, 
                          encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        """Initialize anchor points and their representations."""
        if self.is_fixed_anchor_mode:
            raise RuntimeError("❌ Cannot initialize anchors when using fixed anchor mode.")
        
        # Step 1: Select anchor nodes from original target features
        anchor_features = self.anchor_selector.select_anchors(
            target_features, edge_index, self.num_anchors
        )
        
        logger.info(f"Selected {anchor_features.size(0)} anchors using {type(self.anchor_selector).__name__}")
        
        # Step 2: Map anchors to hidden space
        self.anchor_mapper.initialize(anchor_features, encoder, edge_index)
        logger.info(f"Mapped anchors using {type(self.anchor_mapper).__name__}")
    
    def forward(self, prompted_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss."""
        anchor_representations = self.anchor_mapper.get_anchor_representations()
        
        divergence = self.divergence_metric.compute_divergence(
            prompted_embeddings, anchor_representations
        )
        
        return self.beta * divergence