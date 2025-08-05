"""Base classes for graph neural network models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class GraphEncoder(nn.Module, ABC):
    """Abstract base class for graph encoders.
    
    All graph encoders should inherit from this class and implement
    the forward method. This ensures consistent interfaces across
    different GNN architectures.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.0, **kwargs):
        """Initialize graph encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout rate
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Store additional kwargs for model-specific parameters
        self.model_kwargs = kwargs
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the graph encoder.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch vector for graph-level tasks (optional)
            
        Returns:
            Node embeddings [N, hidden_dim]
        """
        pass
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        return {
            'model_type': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'parameter_count': self.get_parameter_count(),
            'model_kwargs': self.model_kwargs
        }
    
    def log_model_info(self):
        """Log model information."""
        info = self.get_model_info()
        logger.info(f"Model: {info['model_type']}")
        logger.info(f"  Input dim: {info['input_dim']}")
        logger.info(f"  Hidden dim: {info['hidden_dim']}")
        logger.info(f"  Layers: {info['num_layers']}")
        logger.info(f"  Dropout: {info['dropout']}")
        logger.info(f"  Parameters: {info['parameter_count']:,}")
        
        if info['model_kwargs']:
            logger.info("  Additional parameters:")
            for key, value in info['model_kwargs'].items():
                logger.info(f"    {key}: {value}")


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.
    
    Projects node embeddings to a lower-dimensional space for
    contrastive loss computation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 output_dim: Optional[int] = None, num_layers: int = 2,
                 activation: str = 'relu', dropout: float = 0.0):
        """Initialize projection head.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (defaults to hidden_dim)
            num_layers: Number of layers (minimum 1)
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            dropout: Dropout rate
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = hidden_dim
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build layers
        layers = []
        
        if num_layers == 1:
            # Single layer projection
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Multi-layer projection
            # First layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
            ])
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    self._get_activation(activation),
                ])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize parameters
        self.reset_parameters()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def reset_parameters(self):
        """Reset all parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input embeddings [N, input_dim]
            
        Returns:
            Projected embeddings [N, output_dim]
        """
        return self.projection(x)


class Classifier(nn.Module):
    """Classification head for downstream tasks.
    
    Supports both simple linear classification and multi-layer
    classification with optional regularization.
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dim: Optional[int] = None, num_layers: int = 1,
                 activation: str = 'relu', dropout: float = 0.5,
                 batch_norm: bool = False, layer_norm: bool = False):
        """Initialize classifier.
        
        Args:
            input_dim: Input embedding dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension (for multi-layer classifier)
            num_layers: Number of layers (minimum 1)
            activation: Activation function
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        
        if batch_norm and layer_norm:
            raise ValueError("Cannot use both batch_norm and layer_norm")
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build classifier
        if num_layers == 1 or hidden_dim is None:
            # Simple linear classifier
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            # Multi-layer classifier
            layers = []
            current_dim = input_dim
            
            # Hidden layers
            for i in range(num_layers - 1):
                layers.append(nn.Linear(current_dim, hidden_dim))
                
                # Normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                elif layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                
                # Activation
                layers.append(self._get_activation(activation))
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                
                current_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(current_dim, num_classes))
            
            self.classifier = nn.Sequential(*layers)
        
        # Initialize parameters
        self.reset_parameters()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        return activations[activation]
    
    def reset_parameters(self):
        """Reset all parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input embeddings [N, input_dim]
            
        Returns:
            Class logits [N, num_classes]
        """
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """Print detailed model information.
    
    Args:
        model: PyTorch model
        model_name: Name for display
    """
    total_params = count_parameters(model)
    
    logger.info(f"\n📊 {model_name} Information:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Module breakdown
    logger.info("   Module breakdown:")
    for name, module in model.named_children():
        module_params = count_parameters(module)
        logger.info(f"     {name}: {module_params:,} parameters")
    
    # Additional info for graph encoders
    if isinstance(model, GraphEncoder):
        info = model.get_model_info()
        logger.info(f"   Architecture: {info['model_type']}")
        logger.info(f"   Input/Hidden dims: {info['input_dim']}/{info['hidden_dim']}")
        logger.info(f"   Layers: {info['num_layers']}")


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive model summary.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    summary = {
        'model_class': model.__class__.__name__,
        'total_parameters': count_parameters(model),
        'modules': {}
    }
    
    # Module-wise parameter count
    for name, module in model.named_children():
        summary['modules'][name] = {
            'class': module.__class__.__name__,
            'parameters': count_parameters(module)
        }
    
    # Additional info for graph encoders
    if isinstance(model, GraphEncoder):
        summary.update(model.get_model_info())
    
    return summary