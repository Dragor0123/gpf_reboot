import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.nn.inits import glorot, zeros
from typing import Optional


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) Encoder
    """
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
            if i < self.num_layers - 1:  # No dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network (GCN) Encoder
    """
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
    """
    Graph Attention Network (GAT) Encoder
    """
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
    """
    GraphSAGE Encoder
    """
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


def create_model(model_type: str, input_dim: int, hidden_dim: int, 
                num_layers: int = 2, dropout: float = 0.0, **kwargs):
    """
    Create a GNN encoder model.
    
    Args:
        model_type: Type of GNN ('gin', 'gcn', 'gat', 'sage')
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    """
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
    """
    Simple classifier for downstream tasks
    """
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