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
from torch_geometric.utils import negative_sampling
from models import create_model
from datasets import load_dataset


class EdgePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h, edge_index):
        src, dst = edge_index
        edge_feat = torch.cat([h[src], h[dst]], dim=1)
        return self.scorer(edge_feat).squeeze()


def train_edge_prediction(encoder, predictor, data, optimizer, device):
    encoder.train()
    predictor.train()
    data = data.to(device)

    optimizer.zero_grad()
    h = encoder(data.x, data.edge_index)

    pos_edge = data.edge_index
    neg_edge = negative_sampling(
        edge_index=pos_edge, num_nodes=data.num_nodes,
        num_neg_samples=pos_edge.size(1)
    )

    pos_pred = predictor(h, pos_edge)
    neg_pred = predictor(h, neg_edge)

    pos_labels = torch.ones(pos_pred.size(), device=device)
    neg_labels = torch.zeros(neg_pred.size(), device=device)

    loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_pred, neg_pred]),
        torch.cat([pos_labels, neg_labels])
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    config = load_config("config.yaml")
    set_seed(config['experiment'].get('seed', 42))
    device = get_device(config['experiment'].get('device', 'auto'))
    setup_logging()
    log_experiment_info(config)

    # Dataset
    logging.info(f"Loading dataset: {config['dataset']['name']}")
    dataset_info, train_loader, _, _ = load_dataset(config['dataset']['name'])
    data = next(iter(train_loader))

    # Model
    encoder = create_model(
        model_type='gin',
        input_dim=dataset_info['num_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    predictor = EdgePredictor(hidden_dim=config['model']['hidden_dim']).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    logging.info("Start edge prediction pretraining...")
    for epoch in range(config['training']['epochs']):
        loss = train_edge_prediction(encoder, predictor, data, optimizer, device)
        logging.info(f"Epoch {epoch:03d} | Loss: {loss:.4f}")

        # Save every N epochs
        # if epoch % config['training'].get('save_interval') == 0:
        #     save_ckpt(encoder, optimizer, epoch, config, name=f"encoder_epoch_{epoch}", filepath=f"checkpoints/encoder_epoch_{epoch}.pt")
            
    # Final save
    save_ckpt(encoder, optimizer, epoch, config, name="encoder_final", filepath="checkpoints/encoder_final.pt")
    logging.info("Pretraining complete.")


if __name__ == "__main__":
    main()
