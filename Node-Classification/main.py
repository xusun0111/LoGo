import argparse
import math

import torch

from model import build_partition_model
from mtg import evaluate_mtg
from pretrain_losses import mean_balance_loss, neighbor_squared_loss
from utils.get_train_test_data import get_data
from utils.seed import set_seed

def build_parser(preset='homo'):
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group('Experiment and data')
    data.add_argument('--preset', choices=PRESETS, default=preset)
    data.add_argument('--datasets_dir', default=str(Path(__file__).resolve().parent / 'datasets'))
    parser.add_argument('--train_datasets', type=str, nargs='+',default=['CiteSeer', 'PubMed', 'Cornell', 'Squirrel', 'Chameleon'], help='train datasets.')
    parser.add_argument('--test_dataset', type=str,default='Cora', help='test dataset.')
    data.add_argument('--shot_num', type=int, default=1)
    data.add_argument('--mtg_output_dir', default='Experiment/LoGo_node_results')
    train = parser.add_argument_group('Pretraining and model')
    train.add_argument('--epochs', type=int, default=400)
    train.add_argument('--lr', type=float, default=1e-5)
    train.add_argument('--loss_ratio', type=float, default=.1, help='Neighbor-loss weight.')
    train.add_argument('--svd_k', type=int, default=64, help='Unified PCA feature width.')
    train.add_argument('--hidden_dimension', type=int, default=256)
    train.add_argument('--out_dimension', type=int, default=256)
    train.add_argument('--GCN_num1', type=int, default=3, help='First branch depth, at least two.')
    train.add_argument('--GCN_num2', type=int, default=2, help='Second branch depth, at least two.')
    train.add_argument('--top_k', type=int, default=10, help='Neighbors in the feature similarity graph.')
    train.add_argument('--split', type=int, default=4)
    train.add_argument('--joint_kappa', type=float, default=5., help='Global prior weight in joint partitioning.')
    train.add_argument('--DAD', type=int, default=3, help='Propagation hops; -1 selects automatically.')
    adapt = parser.add_argument_group('Anchored adaptation')
    adapt.add_argument('--mtg_steps', type=int, default=50)
    adapt.add_argument('--mtg_lr', type=float, default=1e-4)
    adapt.add_argument('--mtg_anchor', type=float, default=1., help='Positive representation-anchor weight.')
    adapt.add_argument('--mtg_anchor_nodes', type=int, default=1024, help='Anchor nodes per trial; 0 uses all.')
    run = parser.add_argument_group('Runtime')
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.set_defaults(**PRESETS[preset])
    return parser


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--preset', choices=PRESETS, default='homo')
    preset = selector.parse_known_args(argv)[0].preset
    return build_parser(preset).parse_args(argv)


def validate_args(args):
    for name in ('lr', 'mtg_lr', 'mtg_anchor'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(name + ' must be finite and positive')
    for name in ('loss_ratio', 'joint_kappa'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(name + ' must be finite and nonnegative')
    for name in ('epochs', 'mtg_steps', 'mtg_anchor_nodes', 'top_k'):
        if getattr(args, name) < 0:
            raise ValueError(name + ' must be nonnegative')
    for name in ('shot_num', 'num_trials', 'svd_k', 'hidden_dimension', 'out_dimension'):
        if getattr(args, name) < 1:
            raise ValueError(name + ' must be positive')
    if min(args.GCN_num1, args.GCN_num2) < 2:
        raise ValueError('Both GCN branches need at least two layers for last-layer adaptation')
    if args.DAD < -1:
        raise ValueError('DAD must be -1 (automatic) or nonnegative')
    if args.split < 2 or args.svd_k // args.split < 1:
        raise ValueError('split must leave both feature slots nonempty')


def main(args):
    validate_args(args)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    set_seed(args.seed)
    train_dataset, test_data = get_data(args)
    model = build_partition_model(test_data.x.shape[1], args.hidden_dimension,
                                  args.out_dimension, args.GCN_num1, args.GCN_num2, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    train_dataset = [data.to(device) for data in train_dataset]
    for epoch in range(args.epochs):
        model.train()
        for data in train_dataset:
            optimizer.zero_grad(set_to_none=True)
            z1, z2 = model(data.x1, data.edge_index, data.x2, data.edge_index2, args, train=1)
            loss = (mean_balance_loss(z1) + args.loss_ratio * neighbor_squared_loss(z1, data.edge_index)
                    + mean_balance_loss(z2) + args.loss_ratio * neighbor_squared_loss(z2, data.edge_index2))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pretraining loss')
            loss.backward()
            optimizer.step()
        if args.verbose:
            print(f'[LoGo] Epoch {epoch + 1}/{args.epochs}')
    optimizer.zero_grad(set_to_none=True)
    for data in train_dataset:
        data.cpu()
    return evaluate_mtg(model, test_data.to(device), args)


if __name__ == '__main__':
    main(parse_args())
