import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINConv


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, heads=1, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ELU()

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(GINConv(nn1))
        for _ in range(num_layers - 1):
            nn_k = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(GINConv(nn_k))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        return x


def create_model(model_type, input_dim, hidden_dim, num_layers, dropout):
    if model_type == 'gcn':
        return GCNEncoder(input_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'gat':
        return GATEncoder(input_dim, hidden_dim, num_layers, heads=1, dropout=dropout)
    elif model_type == 'gin':
        return GINEncoder(input_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
