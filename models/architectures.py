"""Graph neural network architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, GCN2Conv
from torch_geometric.nn.inits import glorot, zeros
from typing import Optional, Union

from .base import GraphEncoder


class GCNIIEncoder(GraphEncoder):
    """Graph Convolutional Network II (GCNII) Encoder.
    
    Designed for better cross-domain transfer with:
    - Initial residual connections (alpha parameter)
    - Identity mapping (theta parameter) 
    - Over-smoothing prevention
    - Deeper networks capability
    
    Reference: "Simple and Deep Graph Convolutional Networks" (ICML 2020)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 8, 
                 alpha: float = 0.2, theta: float = 1.0, dropout: float = 0.6,
                 layer_norm: bool = True, shared_weights: bool = True):
        """Initialize GCNII encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCNII layers (can be much deeper than GIN)
            alpha: Initial residual connection weight [0, 1]
                  Higher alpha = more initial feature preservation
            theta: Hyperparameter for identity mapping strength
                  Used to compute beta = log(theta/layer + 1)
            dropout: Dropout rate
            layer_norm: Whether to use layer normalization
            shared_weights: Whether to share weights across layers
        """
        super().__init__(input_dim, hidden_dim, num_layers, dropout,
                        alpha=alpha, theta=theta, layer_norm=layer_norm,
                        shared_weights=shared_weights)
        
        self.alpha = alpha
        self.theta = theta
        self.layer_norm = layer_norm
        self.shared_weights = shared_weights
        
        # Initial linear transformation to project to hidden_dim
        self.lin_in = nn.Linear(input_dim, hidden_dim)
        
        # GCNII layers using GCN2Conv
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    channels=hidden_dim,
                    alpha=alpha,
                    theta=theta,
                    layer=layer + 1,  # layer number (1-indexed)
                    shared_weights=shared_weights,
                    cached=False,  # Don't cache for cross-domain
                    add_self_loops=True,
                    normalize=True
                )
            )
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Optional layer normalization for stability
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.lin_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.layer_norms is not None:
            for norm in self.layer_norms:
                norm.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of GCNII encoder.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch vector (unused, for compatibility)
            
        Returns:
            Node embeddings [N, hidden_dim]
        """
        # Initial linear transformation
        x = self.lin_in(x)  # [N, hidden_dim]
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Store initial features for residual connections
        x_0 = x.clone()  # Initial features after first transformation
        
        # GCNII layers
        for i, conv in enumerate(self.convs):
            # GCNII forward: needs both current x and initial x_0
            x = conv(x, x_0, edge_index)
            
            # Layer normalization for stability
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            
            # Activation and dropout (except for last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x


class GINEncoder(GraphEncoder):
    """Graph Isomorphism Network (GIN) Encoder.
    
    GIN is theoretically as powerful as the Weisfeiler-Leman test
    and can distinguish different graph structures effectively.
    
    Reference: "How Powerful are Graph Neural Networks?" (ICLR 2019)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.0, eps: float = 0.0, train_eps: bool = False,
                 batch_norm: bool = True):
        """Initialize GIN encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GIN layers
            dropout: Dropout rate
            eps: Initial epsilon value for GIN aggregation
            train_eps: Whether epsilon is trainable
            batch_norm: Whether to use batch normalization
        """
        super().__init__(input_dim, hidden_dim, num_layers, dropout,
                        eps=eps, train_eps=train_eps, batch_norm=batch_norm)
        
        self.eps = eps
        self.train_eps = train_eps
        self.batch_norm = batch_norm
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        # First layer
        mlp = self._build_mlp(input_dim, hidden_dim)
        self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
        
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = self._build_mlp(hidden_dim, hidden_dim)
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
            
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """Build MLP for GIN layer."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            
            if i < self.num_layers - 1:  # No dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GCNEncoder(GraphEncoder):
    """Graph Convolutional Network (GCN) Encoder.
    
    The classic GCN with localized first-order approximation of
    spectral graph convolutions.
    
    Reference: "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.0, improved: bool = False, 
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True):
        """Initialize GCN encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            improved: Whether to use improved GCN
            cached: Whether to cache normalized adjacency matrix
            add_self_loops: Whether to add self loops
            normalize: Whether to normalize adjacency matrix
        """
        super().__init__(input_dim, hidden_dim, num_layers, dropout,
                        improved=improved, cached=cached, 
                        add_self_loops=add_self_loops, normalize=normalize)
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(
            input_dim, hidden_dim,
            improved=improved, cached=cached,
            add_self_loops=add_self_loops, normalize=normalize
        ))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(
                hidden_dim, hidden_dim,
                improved=improved, cached=cached,
                add_self_loops=add_self_loops, normalize=normalize
            ))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GATEncoder(GraphEncoder):
    """Graph Attention Network (GAT) Encoder.
    
    Uses multi-head attention to learn adaptive node representations
    by attending to different neighbors with different weights.
    
    Reference: "Graph Attention Networks" (ICLR 2018)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.0, heads: int = 8, concat: bool = True,
                 negative_slope: float = 0.2, add_self_loops: bool = True):
        """Initialize GAT encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GAT layers
            dropout: Dropout rate
            heads: Number of attention heads
            concat: Whether to concatenate or average attention heads
            negative_slope: Negative slope for LeakyReLU in attention
            add_self_loops: Whether to add self loops
        """
        super().__init__(input_dim, hidden_dim, num_layers, dropout,
                        heads=heads, concat=concat, negative_slope=negative_slope,
                        add_self_loops=add_self_loops)
        
        self.heads = heads
        self.concat = concat
        
        self.convs = nn.ModuleList()
        
        # Calculate dimensions based on concatenation
        if concat and num_layers > 1:
            head_dim = hidden_dim // heads
            first_output_dim = head_dim * heads
        else:
            head_dim = hidden_dim
            first_output_dim = hidden_dim
        
        # First layer
        self.convs.append(GATConv(
            input_dim, head_dim, heads=heads,
            concat=concat if num_layers > 1 else False,
            dropout=dropout, negative_slope=negative_slope,
            add_self_loops=add_self_loops
        ))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(GATConv(
                first_output_dim, head_dim, heads=heads,
                concat=concat, dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            ))
        
        # Last layer (single head for consistent output dimension)
        if num_layers > 1:
            self.convs.append(GATConv(
                first_output_dim, hidden_dim, heads=1,
                concat=False, dropout=dropout,
                negative_slope=negative_slope,
                add_self_loops=add_self_loops
            ))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.elu(x)  # ELU activation is common for GAT
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class SAGEEncoder(GraphEncoder):
    """GraphSAGE Encoder.
    
    Learns node embeddings by sampling and aggregating features
    from a node's local neighborhood.
    
    Reference: "Inductive Representation Learning on Large Graphs" (NIPS 2017)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.0, aggr: str = 'mean', normalize: bool = False,
                 root_weight: bool = True, bias: bool = True):
        """Initialize SAGE encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of SAGE layers
            dropout: Dropout rate
            aggr: Aggregation method ('mean', 'max', 'add')
            normalize: Whether to apply L2 normalization
            root_weight: Whether to add root node transformation
            bias: Whether to use bias in linear layers
        """
        super().__init__(input_dim, hidden_dim, num_layers, dropout,
                        aggr=aggr, normalize=normalize, root_weight=root_weight,
                        bias=bias)
        
        self.normalize = normalize
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(
            input_dim, hidden_dim, aggr=aggr,
            normalize=normalize, root_weight=root_weight, bias=bias
        ))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(
                hidden_dim, hidden_dim, aggr=aggr,
                normalize=normalize, root_weight=root_weight, bias=bias
            ))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


# Model factory function
def create_model(model_type: str, input_dim: int, hidden_dim: int, 
                num_layers: int = 2, dropout: float = 0.0, **kwargs) -> GraphEncoder:
    """Create a GNN encoder model.
    
    Args:
        model_type: Type of GNN ('gin', 'gcn', 'gat', 'sage', 'gcnii')
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        dropout: Dropout rate
        **kwargs: Additional model-specific arguments
    
    Returns:
        GNN encoder model
        
    Raises:
        ValueError: If model_type is unknown
    """
    model_type = model_type.lower()
    
    model_classes = {
        'gin': GINEncoder,
        'gcn': GCNEncoder,
        'gat': GATEncoder,
        'sage': SAGEEncoder,
        'gcnii': GCNIIEncoder,
    }
    
    if model_type not in model_classes:
        available_models = ', '.join(model_classes.keys())
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available models: {available_models}")
    
    model_class = model_classes[model_type]
    
    try:
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        
        # Log model creation
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Created {model_type.upper()} model with {model.get_parameter_count():,} parameters")
        
        return model
        
    except TypeError as e:
        # Handle invalid kwargs for specific model types
        raise ValueError(f"Invalid arguments for {model_type.upper()} model: {e}")


def get_model_requirements(model_type: str) -> dict:
    """Get model-specific parameter requirements and defaults.
    
    Args:
        model_type: Type of GNN model
        
    Returns:
        Dictionary with parameter information
    """
    requirements = {
        'gin': {
            'required': [],
            'optional': ['eps', 'train_eps', 'batch_norm'],
            'defaults': {'eps': 0.0, 'train_eps': False, 'batch_norm': True}
        },
        'gcn': {
            'required': [],
            'optional': ['improved', 'cached', 'add_self_loops', 'normalize'],
            'defaults': {'improved': False, 'cached': False, 'add_self_loops': True, 'normalize': True}
        },
        'gat': {
            'required': [],
            'optional': ['heads', 'concat', 'negative_slope', 'add_self_loops'],
            'defaults': {'heads': 8, 'concat': True, 'negative_slope': 0.2, 'add_self_loops': True}
        },
        'sage': {
            'required': [],
            'optional': ['aggr', 'normalize', 'root_weight', 'bias'],
            'defaults': {'aggr': 'mean', 'normalize': False, 'root_weight': True, 'bias': True}
        },
        'gcnii': {
            'required': [],
            'optional': ['alpha', 'theta', 'layer_norm', 'shared_weights'],
            'defaults': {'alpha': 0.2, 'theta': 1.0, 'layer_norm': True, 'shared_weights': True}
        }
    }
    
    model_type = model_type.lower()
    if model_type not in requirements:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return requirements[model_type]