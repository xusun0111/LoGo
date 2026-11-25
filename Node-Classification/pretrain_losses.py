import torch
import torch.nn.functional as F


def _normalized_embeddings(z):
    if z.ndim != 2 or z.size(0) == 0 or z.size(1) == 0:
        raise ValueError('Expected nonempty [nodes, features] embeddings')
    if not z.is_floating_point():
        raise ValueError('Expected floating-point embeddings')
    if z.dtype in (torch.float16, torch.bfloat16):
        z = z.float()
    return F.normalize(z, p=2, dim=1)


def mean_balance_loss(z):
    z = _normalized_embeddings(z)
    return z.mean(dim=0).square().sum()


def neighbor_squared_loss(z, edge_index):
    z = _normalized_embeddings(z)
    if edge_index.numel() == 0:
        return z.sum() * 0.0
    src, dst = edge_index
    return (z[src] - z[dst]).square().sum(dim=1).mean()
