import numpy as np
import torch
from Dataset_Load import load_graph_classification_dataset
from utils.partition import pca, x_sort, prepare_partition, mean_pairwise_distance


@torch.no_grad()
def cross_domain_normalization(train_graphs, target_graphs, tau=1., block_size=1024):
    if not train_graphs or not np.isfinite(tau) or tau <= 0:
        raise ValueError('CFN needs source graphs and finite positive tau')
    graphs = train_graphs + target_graphs
    stats = []
    for graph in graphs:
        if not torch.isfinite(graph.x).all():
            raise ValueError('Nonfinite PCA features')
        norm = graph.x.norm(dim=1).mean().item()
        dist = mean_pairwise_distance(graph.x, block_size)
        stats.append((norm, dist))
    valid = [(n, d) for n, d in stats[:len(train_graphs)] if n > 1e-12 and d > 1e-12]
    if valid:
        dist_med = float(np.median([d for n, d in valid]))
        normalized_med = float(np.median([d/n for n, d in valid]))
    for graph, (norm, dist) in zip(graphs, stats):
        if norm <= 1e-12 or dist <= 1e-12 or not valid:
            graph.cfn_scale = 1.
        else:
            factor = np.sqrt((dist_med/normalized_med) / norm)
            graph.cfn_scale = max(float(factor), tau) / norm
        graph.x = graph.x * graph.cfn_scale


def get_data(args):
    if args.test_dataset.upper() in {name.upper() for name in args.train_datasets}:
        raise ValueError('Source datasets must exclude the target dataset')
    sources = [load_graph_classification_dataset(name, args.datasets_dir)
               for name in args.train_datasets]
    targets = load_graph_classification_dataset(args.test_dataset, args.datasets_dir)
    train_graphs = [graph for dataset in sources for graph in dataset]
    with torch.no_grad():
        for graph in train_graphs + targets:
            graph.x = pca(graph.x, args)
        cross_domain_normalization(train_graphs, targets)
        for index, graph in enumerate(train_graphs + targets):
            graph.x = x_sort(graph.x, graph.edge_index, args.split)
            prepare_partition(graph, args, str(index))
    return sources, targets
