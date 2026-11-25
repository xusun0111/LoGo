import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Planetoid, WebKB, WikipediaNetwork


def load_dataset(dataset_name, dataset_dir):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name,
                            transform=T.NormalizeFeatures())
    elif dataset_name in ['Cornell', 'Wisconsin', 'Texas']:
        return WebKB(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    elif dataset_name in ['Chameleon', 'Squirrel']:
        return WikipediaNetwork(
            dataset_dir,
            dataset_name,
            transform=T.NormalizeFeatures())
    elif dataset_name in ['Photo', 'Computers']:
        return Amazon(dataset_dir, dataset_name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'Unsupported node dataset: {dataset_name}')
    return dataset
