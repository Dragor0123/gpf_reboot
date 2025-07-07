import torch
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures


def load_dataset(name, batch_size=128, val_ratio=0.1, test_ratio=0.2, shuffle=True):
    """
    Load standard node classification datasets.
    Returns: dataset_info, train_loader, val_loader, test_loader
    """
    name = name.lower()
    root = './datasets/' + name

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name.capitalize(), transform=NormalizeFeatures())
    elif name == 'amazonphoto':
        dataset = Amazon(root=root, name='Photo', transform=NormalizeFeatures())
    elif name == 'amazoncomputers':
        dataset = Amazon(root=root, name='Computers', transform=NormalizeFeatures())
    elif name == 'actor':
        dataset = Actor(root=root, transform=NormalizeFeatures())
    elif name == 'chameleon' or name == 'squirrel':
        dataset = WikipediaNetwork(root=root, name=name.capitalize(), transform=NormalizeFeatures())
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = dataset[0]
    num_nodes = data.num_nodes
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # Split indices
    indices = torch.randperm(num_nodes) if shuffle else torch.arange(num_nodes)
    val_size = int(num_nodes * val_ratio)
    test_size = int(num_nodes * test_ratio)
    train_size = num_nodes - val_size - test_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    # Construct loaders (wrap in lists for compatibility)
    train_loader = DataLoader([data], batch_size=1)
    val_loader = DataLoader([data], batch_size=1)
    test_loader = DataLoader([data], batch_size=1)

    dataset_info = {
        'num_features': num_features,
        'num_classes': num_classes
    }

    return dataset_info, train_loader, val_loader, test_loader
