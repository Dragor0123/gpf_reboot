import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot


class GPFPrompt(nn.Module):
    def __init__(self, input_dim: int, p_num: int):
        super(GPFPrompt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, input_dim))
        self.a = nn.Linear(input_dim, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    

class ResidualMLPPrompt(nn.Module):
    """
    Expressive, MMD-aligned prompt module: x + MLP(x)
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        last_dim = input_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else input_dim
            layers.append(nn.Linear(last_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            last_dim = next_dim
        self.mlp = nn.Sequential(*layers)
        # Optional: LayerNorm after prompt
        self.norm = nn.LayerNorm(input_dim)
        
    def add(self, x):
        # x: [N, F]
        prompt_vec = self.mlp(x)  # shape [N, F]
        #return x + prompt_vec
        return self.norm(x + prompt_vec)  # Residual + Norm
    

class LinearPrompt(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super(LinearPrompt, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.prompt_matrix = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prompt_matrix.weight)
    
    def add(self, x: Tensor):
        return self.prompt_matrix(x)
        