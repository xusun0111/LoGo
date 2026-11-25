import numpy as np
import torch
from sklearn.decomposition import PCA
from torch_geometric.utils import to_undirected
from torch.nn import functional as F

def edge_create(x, args):
    k = min(args.top_k, max(x.size(0) - 1, 0))
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    normalized = torch.nn.functional.normalize(x, dim=1)
    similarities = normalized @ normalized.T
    similarities.fill_diagonal_(-torch.inf)
    neighbors = similarities.topk(k, dim=1).indices
    rows = torch.arange(x.size(0), device=x.device)[:, None].expand(-1, k)
    return torch.stack([rows.reshape(-1), neighbors.reshape(-1)])

@torch.no_grad()
def mean_pairwise_distance(x, block_size=1024):
    if x.ndim != 2 or x.size(0) == 0:
        raise ValueError('Expected a non-empty node feature matrix.')
    if block_size <= 0:
        raise ValueError('block_size must be positive.')
    n = x.size(0)
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            distances = torch.cdist(
                x[i:i + block_size], x[j:j + block_size], p=2
            )
            if i == j:
                distances.fill_diagonal_(0)
            total += distances.sum(dtype=torch.float64)
    return (total / (n * n)).item()

def effective_partition(x, split):
    if split < 2 or x.size(1) // split < 1:
        raise ValueError('split must leave both input slots nonempty')
    active = torch.nonzero(x.ne(0).any(dim=0), as_tuple=False).flatten()
    count = active.numel()
    quota = min(max(count // split, 1), count - 1) if count >= 2 else count
    return active, quota


def x_sort(x, edge_index, split=4):
    active, quota = effective_partition(x, split)
    score = feature_homophily_scores(x, edge_index)
    ordered = active[torch.argsort(score[active], stable=True)]
    mid = x.size(1) // split
    result = torch.zeros_like(x)
    result[:, :quota] = x[:, ordered[:quota]]
    result[:, mid:mid + active.numel() - quota] = x[:, ordered[quota:]]
    return result

def feature_homophily_scores(x, edge_index):
    src, dst = to_undirected(edge_index)
    return (((x[src] - x[dst]) ** 2).mean(dim=0) if src.numel() else x.new_zeros(x.size(1)))

def pca(x, args):
    if x.size(1) < args.svd_k:
        x = F.pad(x, (0, args.svd_k + 2 - x.size(1)), value=0)
    x = x.detach().cpu().numpy()
    n_samples, n_features = x.shape
    if n_features < args.svd_k:
        x = np.pad(x, ((0,0), (0,args.svd_k-n_features)))
    if n_samples < args.svd_k:
        x = np.pad(x, ((0,args.svd_k-n_samples), (0,0)))
    result = PCA(n_components=args.svd_k, random_state=args.seed).fit_transform(x)
    return torch.tensor(result[:n_samples], dtype=torch.float)


@torch.no_grad()
def joint_hard_mask(x, edge_index, split=4, kappa=5.0):
    if x.ndim != 2 or x.size(0) == 0 or x.size(1) == 0:
        raise ValueError('Expected nonempty node features')
    if not x.is_floating_point() or not torch.isfinite(x).all().item():
        raise ValueError('Expected finite floating point features')
    if split < 2 or x.size(1) // split < 1:
        raise ValueError('split must leave both branches nonempty')
    if np.isnan(kappa) or kappa < 0:
        raise ValueError('joint_kappa must be nonnegative (positive infinity is allowed)')
    active, quota = effective_partition(x, split)
    global_score = feature_homophily_scores(x, edge_index)
    ordered = active[torch.argsort(global_score[active], stable=True)]
    if active.numel() >= 2:
        low, high = global_score[ordered[quota-1]], global_score[ordered[quota]]
        threshold = low + (high-low) * 0.5
    else:
        threshold = global_score[active[0]] if active.numel() else x.new_zeros(())
    valid = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)
    valid[active] = True
    if active.numel() <= 1:
        return valid.unsqueeze(0).expand_as(x).clone(), threshold
    if np.isinf(kappa):
        mask = ((global_score <= threshold) & valid).unsqueeze(0).expand_as(x).clone()
        return mask, threshold
    src, dst = to_undirected(edge_index, num_nodes=x.size(0))
    degree = torch.bincount(src, minlength=x.size(0)).to(x.dtype)
    sums = torch.zeros_like(x)
    for start in range(0, src.numel(), 8192):
        u, v = src[start:start+8192], dst[start:start+8192]
        sums.index_add_(0, u, (x[u] - x[v]).square())
    local = sums / degree.clamp_min(1)[:, None]
    local_weight = degree / (degree + kappa).clamp_min(torch.finfo(x.dtype).tiny)
    joint_score = (1-local_weight[:, None])*global_score + local_weight[:, None]*local
    mask = (joint_score <= threshold) & valid.unsqueeze(0)
    return mask, threshold

@torch.no_grad()
def prepare_partition(data, args, label='graph'):
    if args.split < 2 or data.x.size(1) // args.split < 1:
        raise ValueError('split must leave both branches nonempty')
    mid = data.x.size(1) // args.split
    data.edge_index2 = edge_create(data.x[:, mid:], args)
    mask, threshold = joint_hard_mask(data.x, data.edge_index, args.split, args.joint_kappa)
    data.x1 = data.x * mask
    data.x2 = data.x * ~mask
    if args.verbose:
        print(f'[JointPartition] {label}: threshold={threshold.item():.6g}, '
              f'first_branch_fraction={mask.float().mean().item():.2%}')
