import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DualGCN(nn.Module):
    def __init__(self, in1, in2, hidden_dim, out_dim, GCN_num1, GCN_num2):
        super(DualGCN, self).__init__()

        self.gcn1 = GCN(in1, hidden_dim, out_dim, GCN_num1)
        self.gcn2 = GCN(in2, hidden_dim, out_dim, GCN_num2)


    def forward(self, x1, edge_index1, x2, edge_index2, args, train = 1, dad_operator=None):
        out1 = self.gcn1(x1, edge_index1)
        out2 = self.gcn2(x2, edge_index2)
        if train == 1:
            return out1, out2
        else:
            z = torch.cat([out1, out2], dim=1)
            num_hop = args.DAD
            if num_hop != 0:
                a = dad_operator if dad_operator is not None else sparse_dad(edge_index1, z.size(0), z.dtype)
                for i in range(num_hop):
                    z = torch.sparse.mm(a, z) if a.is_sparse else a @ z
            return z

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_dimension, out_channels, k = 2):
        super(GCN, self).__init__()
        self.layer = nn.ModuleList()
        self.layer.append(GCNConv(in_channels, hidden_dimension))
        for _ in range(1,k-1):
            self.layer.append(GCNConv(hidden_dimension, hidden_dimension))
        if k > 1:
            self.layer.append(GCNConv(hidden_dimension, out_channels))
        self.act = nn.ReLU()


    def forward(self, x: torch.Tensor,edge_index: torch.Tensor, train=1) -> torch.Tensor:
        adapters = getattr(self, 'mtg', None)
        for i, conv in enumerate(self.layer):
            if adapters is not None and str(i) in adapters:
                x = adapters[str(i)](x)
            x = conv(x, edge_index)
            if i < len(self.layer) - 1:
                x = self.act(x)
        return x


def sparse_dad(edge_index, num_nodes, dtype):
    loops = torch.arange(num_nodes, device=edge_index.device).repeat(2, 1)
    indices = torch.cat([edge_index, loops], dim=1)
    values = torch.ones(indices.size(1), dtype=dtype, device=edge_index.device)
    a = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    row, col = a.indices()
    degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
    degree.index_add_(0, col, a.values())
    inv = degree.rsqrt()
    return torch.sparse_coo_tensor(a.indices(), a.values() * inv[row] * inv[col],
                                   a.shape).coalesce()


def build_partition_model(feature_dim, hidden_dim, out_dim, layers1, layers2, args):
    mid = feature_dim // args.split
    if args.split < 2 or mid < 1:
        raise ValueError('split must leave both branches nonempty')
    reference = DualGCN(mid, feature_dim-mid, hidden_dim, out_dim, layers1, layers2)
    with torch.random.fork_rng(devices=[]):
        expanded = DualGCN(feature_dim, feature_dim, hidden_dim, out_dim, layers1, layers2)
    state = expanded.state_dict()
    for name, value in reference.state_dict().items():
        if name == 'gcn1.layer.0.lin.weight':
            state[name].zero_()
            state[name][:, :mid].copy_(value)
        elif name == 'gcn2.layer.0.lin.weight':
            state[name].zero_()
            state[name][:, mid:].copy_(value)
        else:
            state[name].copy_(value)
    expanded.load_state_dict(state)
    return expanded
