import torch

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
