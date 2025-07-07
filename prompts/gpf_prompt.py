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