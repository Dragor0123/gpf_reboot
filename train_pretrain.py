import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    set_seed, get_device, load_config,
    setup_logging, log_experiment_info,
    save_ckpt
)
import logging
from models import create_model
from datasets import load_dataset
from torch_geometric.data import Data


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.proj(x)


def graph_views(data: Data, aug: str = 'dropN', aug_ratio: float = 0.2):
    x, edge_index = data.x.clone(), data.edge_index.clone()

    if aug == 'dropN':  # node feature masking (drop node features)
        mask = torch.rand(x.size(0), device=x.device) > aug_ratio
        x = x * mask.unsqueeze(1).float()

    elif aug == 'permE':  # edge permutation (randomly drop edges)
        num_edges = edge_index.size(1)
        keep = torch.rand(num_edges, device=edge_index.device) > aug_ratio
        edge_index = edge_index[:, keep]

    elif aug == 'maskN':  # feature masking (random noise)
        noise = torch.randn_like(x) * aug_ratio
        x = x + noise

    elif aug == 'dropE':  # explicitly drop edges without permuting
        num_edges = edge_index.size(1)
        drop = torch.rand(num_edges, device=edge_index.device) < aug_ratio
        edge_index = edge_index[:, ~drop]

    elif aug == 'maskF':  # hard masking of random features
        mask = torch.rand_like(x) > aug_ratio
        x = x * mask

    return Data(x=x, edge_index=edge_index)


def info_nce_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.t())
    positives = torch.diag(sim_matrix)
    numerator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(sim_matrix / temperature), dim=1)
    loss = -torch.log(numerator / denominator)
    return loss.mean()


def main():
    config = load_config("config.yaml")
    set_seed(config['experiment'].get('seed', 42))
    device = get_device(config['experiment'].get('device', 'auto'))
    setup_logging()
    log_experiment_info(config)

    # Dataset
    logging.info(f"Loading dataset: {config['dataset']['name']}")
    dataset_info, train_loader, _, _ = load_dataset(config['dataset']['name'])
    data = next(iter(train_loader)).to(device)

    # Model + Projection Head
    encoder = create_model(
        model_type=config['model']['type'],
        input_dim=dataset_info['num_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    projector = ProjectionHead(
        input_dim=config['model']['hidden_dim'],
        hidden_dim=config['model']['hidden_dim']
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=config['pretrain']['lr'],
        weight_decay=config['pretrain']['weight_decay']
    )

    aug1 = config['augmentation']['view1']
    aug2 = config['augmentation']['view2']
    aug_ratio = config['augmentation']['aug_ratio']

    logging.info("Start GCL-style contrastive pretraining...")
    for epoch in range(config['pretrain']['epochs']):
        encoder.train()
        projector.train()

        view1 = graph_views(data, aug=aug1, aug_ratio=aug_ratio)
        view2 = graph_views(data, aug=aug2, aug_ratio=aug_ratio)

        h1 = encoder(view1.x.to(device), view1.edge_index.to(device))
        h2 = encoder(view2.x.to(device), view2.edge_index.to(device))

        z1 = projector(h1)
        z2 = projector(h2)

        loss = info_nce_loss(z1, z2, temperature=0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info(f"Epoch {epoch:03d} | Contrastive Loss: {loss:.4f}")

    # Save final encoder
    save_ckpt(encoder, optimizer, epoch, loss.item(), "checkpoints/encoder_final.pt")
    logging.info("GCL Pretraining complete.")


if __name__ == "__main__":
    main()
