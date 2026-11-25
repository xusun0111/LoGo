from Dataset_Load import load_dataset
from torch_geometric.utils import to_undirected
from sklearn.decomposition import PCA
import numpy as np
import torch

def get_data(args):
    with torch.no_grad():
        train_dataset = []
        for dataset_name in args.train_datasets:
            train_data = load_dataset(dataset_name, args.datasets_dir)[0]
            if getattr(args, 'verbose', False):
                print(train_data)
            if getattr(args, 'verbose', False):
                print(torch.max(train_data.y).item() + 1)
            train_data.x = pca(train_data.x, args)
            train_dataset.append(train_data)

        test_data = load_dataset(args.test_dataset, args.datasets_dir)[0]
        test_data.x = pca(test_data.x, args)

        train_dataset, test_data = cross_domain_normalization(
            train_dataset, test_data, verbose=args.verbose)

        for dataset_name, train_data in zip(args.train_datasets, train_dataset):
            train_data.x = x_sort(train_data.x, train_data.edge_index)
            prepare_partition(train_data, args, 'train:' + dataset_name)
            train_data.detach().cpu()

        test_data.x = x_sort_test(
            test_data.x, test_data.edge_index, args,
            total_scale=getattr(test_data, "cfn_scale", 1.0),
        )
        prepare_partition(test_data, args, 'test:' + args.test_dataset)
        test_data.detach().cpu()
    return train_dataset, test_data


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


@torch.no_grad()
def cross_domain_normalization(
    train_dataset, test_data, tau=1.0, block_size=1024, eps=1e-12, verbose=False
):
    if not train_dataset:
        raise ValueError('At least one training graph is required.')
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive.')
    graphs = list(train_dataset) + [test_data]
    stats = []
    for index, graph in enumerate(graphs):
        x = graph.x
        if x.ndim != 2 or x.size(0) == 0:
            raise ValueError(f'Graph {index}: empty node feature matrix.')
        if not torch.isfinite(x).all().item():
            raise ValueError(f'Graph {index}: features contain NaN or Inf.')
        norm = torch.norm(x, p=2, dim=1).mean().item()
        if not np.isfinite(norm) or norm <= eps:
            raise ValueError(f'Graph {index}: mean feature norm is too small.')
        dist = mean_pairwise_distance(x, block_size)
        if not np.isfinite(dist) or dist <= eps:
            raise ValueError(f'Graph {index}: mean pairwise distance is too small.')
        stats.append((norm, dist, dist / norm))

    train_stats = stats[:len(train_dataset)]
    dist_med = float(np.median([s[1] for s in train_stats]))
    dist_normalized_med = float(np.median([s[2] for s in train_stats]))
    for index, (graph, (norm, dist, dist_normalized)) in enumerate(zip(graphs, stats)):
        f = np.sqrt(
            (dist_med / dist_normalized_med) * (dist_normalized / dist)
        )
        graph.cfn_scale = max(float(f), float(tau)) / norm
        graph.x = graph.x * graph.cfn_scale
        label = f'train[{index}]' if index < len(train_dataset) else 'test'
        if verbose:
            print(f'[CFN] {label}: mean_norm={norm:.6g}, f={f:.6g}, '
                  f'total_scale={graph.cfn_scale:.6g}')
    return train_dataset, test_data


def edge_create(x, args):

    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    sim_matrix = torch.matmul(x_norm, x_norm.t())

    num_nodes = x.size(0)
    topk = args.top_k

    _, topk_indices = torch.topk(sim_matrix, k=topk + 1, dim=-1)
    topk_indices = topk_indices[:, 1:]

    row_indices = torch.arange(num_nodes).unsqueeze(1).expand(-1, topk).reshape(-1)
    col_indices = topk_indices.reshape(-1)
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    return edge_index

def x_sort(x, edge_index):
    edge_index = to_undirected(edge_index)

    src, dst = edge_index
    feat_diff = x[src] - x[dst]

    feature_similarity = ((feat_diff ** 2).mean(dim=0) if src.numel() else x.new_zeros(x.size(1)))

    sorted_indices = torch.argsort(feature_similarity)

    x_reordered = x[:, sorted_indices]

    return x_reordered
def feature_homophily_scores(x, edge_index):
    src, dst = to_undirected(edge_index)
    return (((x[src] - x[dst]) ** 2).mean(dim=0) if src.numel() else x.new_zeros(x.size(1)))


def x_sort_test(x, edge_index, args, total_scale=1.0):
    feature_similarity = feature_homophily_scores(x, edge_index)
    range_val = (feature_similarity.max() - feature_similarity.min()).item()

    threshold = 0.01 * total_scale ** 2
    automatic = args.DAD == -1
    if automatic:
        args.DAD = 0 if range_val > threshold else 5

    if getattr(args, 'verbose', False):
        print(f'[DAD] input_range={range_val:.6g}, '
              f'total_scale={total_scale:.6g}, threshold={threshold:.6g}, '
              f'selected={args.DAD}, automatic={automatic}')
    sorted_indices = torch.argsort(feature_similarity)
    return x[:, sorted_indices]


def pca(x, args):
    x = x.detach().cpu().numpy()
    n_samples, n_features = x.shape
    if min(n_samples, n_features) == 0 or args.svd_k <= 0:
        raise ValueError('PCA requires non-empty features and positive svd_k.')
    n_components = min(args.svd_k, n_samples, n_features)
    projector = PCA(n_components=n_components, random_state=args.seed)
    result = projector.fit_transform(x)
    if n_components < args.svd_k:
        result = np.pad(result, ((0, 0), (0, args.svd_k - n_components)),
                        mode='constant', constant_values=0)
    return torch.tensor(result, dtype=torch.float)


@torch.no_grad()
def joint_hard_mask(x, edge_index, split=4, kappa=5.0):
    if x.ndim != 2 or x.size(0) == 0 or x.size(1) == 0:
        raise ValueError('Expected nonempty node features')
    if not x.is_floating_point() or not torch.isfinite(x).all().item():
        raise ValueError('Expected finite floating point features')
    if split < 2 or x.size(1) // split < 1:
        raise ValueError('split must leave both branches nonempty')
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError('joint_kappa must be finite and nonnegative')
    mid = x.size(1) // split
    global_mask = torch.zeros_like(x, dtype=torch.bool)
    global_mask[:, :mid] = True
    global_score = feature_homophily_scores(x, edge_index)
    threshold = global_score[mid-1] + (global_score[mid] - global_score[mid-1]) * 0.5
    src, dst = to_undirected(edge_index, num_nodes=x.size(0))
    degree = torch.bincount(src, minlength=x.size(0)).to(x.dtype)
    sums = torch.zeros_like(x)
    for start in range(0, src.numel(), 8192):
        u, v = src[start:start+8192], dst[start:start+8192]
        sums.index_add_(0, u, (x[u] - x[v]).square())
    observed = degree > 0
    local = sums / degree.clamp_min(1)[:, None]
    local_weight = degree / (degree + kappa).clamp_min(torch.finfo(x.dtype).tiny)
    joint_score = (1-local_weight[:, None])*global_score + local_weight[:, None]*local
    mask = global_mask.clone()
    mask[global_mask & (joint_score > threshold)] = False
    mask[~global_mask & (joint_score < threshold)] = True
    mask[~observed] = global_mask[~observed]
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
