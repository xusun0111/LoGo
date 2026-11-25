import argparse
import math
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from model import build_partition_model
from mtg_graph import evaluate
from pretrain_losses import mean_balance_loss, neighbor_squared_loss
from utils.get_train_test_data import get_data
from utils.seed import set_seed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group('Data and output')
    data.add_argument('--datasets_dir', default=str(Path(__file__).resolve().parent / 'datasets'))
    data.add_argument('--train_datasets', nargs='+', default=['IMDB-BINARY', 'DD'])
    data.add_argument('--test_dataset', default='ENZYMES')
    data.add_argument('--mtg_output_dir', default='Experiment/LoGo_graph_results')
    train = parser.add_argument_group('Pretraining and model')
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--lr', type=float, default=1e-6)
    train.add_argument('--batch_size', type=int, default=128)
    train.add_argument('--svd_k', type=int, default=512, help='Unified PCA feature width.')
    train.add_argument('--hidden_dimension', type=int, default=64)
    train.add_argument('--out_dimension', type=int, default=256)
    train.add_argument('--GCN_num', type=int, default=2, help='GCN depth; at least two layers.')
    train.add_argument('--loss_ratio', type=float, default=.1, help='Neighbor-loss weight.')
    train.add_argument('--top_k', type=int, default=1, help='Neighbors in the feature similarity graph.')
    train.add_argument('--split', type=int, default=4, help='Global feature partition ratio.')
    train.add_argument('--joint_kappa', type=float, default=5., help='Global prior weight in joint partitioning.')
    train.add_argument('--DAD', type=int, default=0, help='Post-encoding propagation hops.')
    adapt = parser.add_argument_group('Anchored adaptation')
    adapt.add_argument('--num_trials', type=int, default=50)
    adapt.add_argument('--mtg_steps', type=int, default=50)
    adapt.add_argument('--mtg_lr', type=float, default=1e-4)
    adapt.add_argument('--mtg_anchor', type=float, default=1., help='Positive representation-anchor weight.')
    adapt.add_argument('--mtg_anchor_graphs', type=int, default=1024, help='Anchor graphs per trial; 0 uses all graphs.')
    run = parser.add_argument_group('Runtime')
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser


def validate_args(args):
    for name in ('lr', 'mtg_lr', 'mtg_anchor'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(name + ' must be finite and positive')
    if not math.isfinite(args.loss_ratio) or args.loss_ratio < 0:
        raise ValueError('loss_ratio must be finite and nonnegative')
    for name in ('epochs', 'mtg_steps', 'DAD', 'mtg_anchor_graphs', 'top_k'):
        if getattr(args, name) < 0:
            raise ValueError(name + ' must be nonnegative')
    for name in ('batch_size', 'svd_k', 'hidden_dimension', 'out_dimension', 'num_trials'):
        if getattr(args, name) < 1:
            raise ValueError(name + ' must be positive')
    if args.GCN_num < 2:
        raise ValueError('GCN_num must be at least two for last-layer adaptation')
    if args.split < 2 or args.svd_k // args.split < 1:
        raise ValueError('split must leave both feature slots nonempty')
    if not math.isfinite(args.joint_kappa) or args.joint_kappa < 0:
        raise ValueError('joint_kappa must be finite and nonnegative')


def main(args):
    validate_args(args)
    set_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    sources, targets = get_data(args)
    model = build_partition_model(args.svd_k, args.hidden_dimension, args.out_dimension,
                                  args.GCN_num, args.GCN_num, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        loaders = [DataLoader(graphs, batch_size=args.batch_size, shuffle=True) for graphs in sources]
        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                z1, z2 = model(batch.x1, batch.edge_index, batch.x2, batch.edge_index2, args, train=1)
                loss = (mean_balance_loss(z1) + args.loss_ratio * neighbor_squared_loss(z1, batch.edge_index)
                        + mean_balance_loss(z2) + args.loss_ratio * neighbor_squared_loss(z2, batch.edge_index2))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite pretraining loss')
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        if args.verbose:
            print(f'Epoch {epoch+1}/{args.epochs}: loss={sum(losses)/len(losses):.6f}')
    optimizer.zero_grad(set_to_none=True)
    return evaluate(model, targets, args)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    main(args)
