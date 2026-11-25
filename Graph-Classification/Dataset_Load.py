import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, add_self_loops, remove_self_loops


def load_graph_classification_dataset(dataset_name, dataset_dir):
    dataset = TUDataset(root=dataset_dir, name=dataset_name.upper())
    graphs = [g.clone() for g in dataset]
    if not graphs or any(g.num_nodes == 0 for g in graphs):
        raise ValueError('Datasets must contain nonempty graphs')
    if graphs[0].x is None:
        use_labels = all(getattr(g, 'node_labels', None) is not None for g in graphs)
        values = [g.node_labels.long().reshape(-1) if use_labels else
                  degree(g.edge_index[0], num_nodes=g.num_nodes).long() for g in graphs]
        width = max(int(v.max()) for v in values) + 1
        for graph, value in zip(graphs, values):
            graph.x = F.one_hot(value, num_classes=width).float()
    for graph in graphs:
        graph.x = graph.x.float()
        graph.edge_index, _ = remove_self_loops(graph.edge_index)
        graph.edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    return graphs
