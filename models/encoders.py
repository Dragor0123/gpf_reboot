import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, GCN2Conv, global_mean_pool
from torch_geometric.nn.inits import glorot, zeros
from typing import Optional


class GCNIIEncoder(nn.Module):
    """
    Graph Convolutional Network II (GCNII) Encoder using GCN2Conv
    
    Designed for better cross-domain transfer with:
    - Initial residual connections (alpha parameter)
    - Identity mapping (theta parameter) 
    - Over-smoothing prevention
    - Deeper networks capability
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 8, 
                 alpha: float = 0.2, theta: float = 1.0, dropout: float = 0.6):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCNII layers (can be much deeper than GIN)
            alpha: Initial residual connection weight [0, 1]
                  Higher alpha = more initial feature preservation
            theta: Hyperparameter for identity mapping strength
                  Used to compute beta = log(theta/layer + 1)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta
        self.dropout = dropout
        
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
                    shared_weights=True,  # Use shared weights (GCNII)
                    cached=False,  # Don't cache for cross-domain
                    add_self_loops=True,
                    normalize=True
                )
            )
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Optional: Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        self.lin_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.layer_norms:
            norm.reset_parameters()
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of GCNII encoder
        
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
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            # GCNII forward: needs both current x and initial x_0
            x = conv(x, x_0, edge_index)
            
            # Layer normalization for stability
            x = norm(x)
            
            # Activation and dropout (except for last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x


class GCNIIEncoder_Simple(nn.Module):
    """
    Simplified GCNII Encoder without layer normalization
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 8, 
                 alpha: float = 0.2, theta: float = 1.0, dropout: float = 0.6):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial transformation
        self.lin_in = nn.Linear(input_dim, hidden_dim)
        
        # GCNII layers
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    channels=hidden_dim,
                    alpha=alpha,
                    theta=theta,
                    layer=layer + 1,
                    shared_weights=True
                )
            )
    
    def forward(self, x, edge_index, batch=None):
        # Initial transformation
        x = F.relu(self.lin_in(x))
        x_0 = x.clone()  # Store initial features
        
        # GCNII layers
        for i, conv in enumerate(self.convs):
            x = conv(x, x_0, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


# Update the create_model function to include GCNII
def create_model(model_type: str, input_dim: int, hidden_dim: int, 
                num_layers: int = 2, dropout: float = 0.0, **kwargs):
    """
    Create a GNN encoder model.
    
    Args:
        model_type: Type of GNN ('gin', 'gcn', 'gat', 'sage', 'gcnii')
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        dropout: Dropout rate
        **kwargs: Additional model-specific arguments
    
    Returns:
        GNN encoder model
    """
    model_type = model_type.lower()
    
    if model_type == 'gin':
        return GINEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'gcn':
        return GCNEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'gat':
        heads = kwargs.get('heads', 8)
        return GATEncoder(input_dim, hidden_dim, num_layers, dropout, heads)
    elif model_type == 'sage':
        return SAGEEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'gcnii':
        # GCNII specific parameters
        alpha = kwargs.get('alpha', 0.2)
        theta = kwargs.get('theta', 1.0)
        # Use simple version by default, can switch to full version
        return GCNIIEncoder_Simple(input_dim, hidden_dim, num_layers, alpha, theta, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Existing encoders remain the same...
class GINEncoder(nn.Module):
    """Graph Isomorphism Network (GIN) Encoder"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNEncoder(nn.Module):
    """Graph Convolutional Network (GCN) Encoder"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network (GAT) Encoder"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 dropout: float = 0.0, heads: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        
        # Last layer (single head)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGEEncoder(nn.Module):
    """GraphSAGE Encoder"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim
            
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Classifier(nn.Module):
    """Simple classifier for downstream tasks"""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        
        if hidden_dim is None:
            # Simple linear classifier
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            # MLP classifier
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, x):
        return self.classifier(x)