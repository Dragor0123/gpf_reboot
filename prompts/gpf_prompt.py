import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot

class GPFPrompt(nn.Module):
    """
    GPF-Reboot 스타일의 additive prompt.
    각 노드에 대해 soft attention으로 prompt vector를 조합하여 더함.
    """
    def __init__(self, input_dim: int, num_prompts: int):
        super(GPFPrompt, self).__init__()
        self.input_dim = input_dim
        self.num_prompts = num_prompts

        self.prompt_vectors = nn.Parameter(torch.Tensor(num_prompts, input_dim))
        self.attention = nn.Linear(input_dim, num_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.prompt_vectors)
        self.attention.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, input_dim] shaped node features
        Returns:
            Tensor: Prompted node features, same shape [N, input_dim]
        """
        score = self.attention(x)                      # [N, P]
        weight = F.softmax(score, dim=1)               # [N, P]
        prompt = weight @ self.prompt_vectors          # [N, D]
        return x + prompt