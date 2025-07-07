import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    set_seed, get_device, load_config,
    setup_logging, log_experiment_info
)
import logging
from models import create_model
from datasets import load_dataset
from prompts.gpf_prompt import GPFPrompt


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def main():
    config = load_config("config.yaml")
    set_seed(config['experiment'].get('seed', 42))
    device = get_device(config['experiment'].get('device', 'auto'))
    setup_logging()
    log_experiment_info(config)

    # Dataset
    dataset_info, train_loader, val_loader, test_loader = load_dataset(config['dataset']['name'])
    data = next(iter(train_loader)).to(device)
    raw_feature_dim = dataset_info['num_features']
    # Load pretrained encoder (frozen)
    encoder = create_model(
        model_type=config['model']['type'],
        input_dim=raw_feature_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    encoder.load_state_dict(torch.load("checkpoints/encoder_final.pt", map_location=device)['model_state_dict'])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Prompt and Classifier
    prompt = GPFPrompt(
        input_dim=raw_feature_dim,
        p_num=config['prompt']['num_prompts']
    ).to(device)

    classifier = Classifier(
        input_dim=config['model']['hidden_dim'],
        num_classes=dataset_info['num_classes']
    ).to(device)

    optimizer = torch.optim.Adam(
        list(prompt.parameters()) + list(classifier.parameters()),
        lr=config['prompt_tuning']['lr'],
        weight_decay=config['prompt_tuning']['weight_decay']
    )

    # Training loop (CE loss only for now)
    for epoch in range(config['prompt_tuning']['epochs']):
        prompt.train()
        classifier.train()

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        mask = data.train_mask

        prompted_x = prompt.add(x)
        with torch.no_grad():
            h = encoder(prompted_x, edge_index)

        out = classifier(h)
        loss = F.cross_entropy(out[mask], y[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info(f"Epoch {epoch:03d} | Prompt Tuning Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
