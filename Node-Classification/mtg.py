import copy
import csv
import json
from datetime import datetime
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from model import sparse_dad


class MessageAdapter(nn.Module):
    def __init__(self, width, num_vectors=2):
        super().__init__()
        self.projection = nn.Linear(width, num_vectors)
        self.vectors = nn.Parameter(torch.zeros(num_vectors, width))

    def forward(self, x):
        return x + self.projection(x).softmax(dim=-1) @ self.vectors


def adapted_copy(backbone):
    model = copy.deepcopy(backbone)
    for param in model.parameters():
        param.requires_grad_(False)
        param.grad = None
    for branch_id in ('1', '2'):
        branch = getattr(model, 'gcn' + branch_id)
        branch.mtg = nn.ModuleDict()
        if len(branch.layer) < 2:
            raise ValueError('Last-layer MTG requires at least two GCN layers')
        i = len(branch.layer) - 1
        conv = branch.layer[i]
        ref = next(conv.parameters())
        branch.mtg[str(i)] = MessageAdapter(conv.in_channels).to(device=ref.device, dtype=ref.dtype)
    return model


def prototypes(embeddings, labels, classes):
    return torch.stack([embeddings[labels == c].mean(0) for c in classes])


class CosineHead(nn.Module):
    def __init__(self, centers, temperature=0.2):
        super().__init__()
        self.weight = nn.Parameter(F.normalize(centers.detach().clone(), dim=-1))
        self.temperature = temperature

    def forward(self, z):
        return F.normalize(z, dim=-1) @ F.normalize(self.weight, dim=-1).T / self.temperature


def encode(model, data, args, dad):
    return model(data.x1, data.edge_index, data.x2, data.edge_index2,
                 args, train=0, dad_operator=dad)


def fit_mtg_trial(backbone, data, z0, support, labels, query, args, dad):
    classes, targets = torch.unique(labels, sorted=True, return_inverse=True)
    if classes.numel() < 2:
        raise ValueError('At least two support classes required')
    head = CosineHead(prototypes(z0[support], labels, classes))
    head.requires_grad_(False)
    model = adapted_copy(backbone).eval()
    count = args.mtg_anchor_nodes
    anchor_idx = (torch.arange(z0.size(0), device=z0.device) if count == 0 or count >= z0.size(0)
                  else torch.randperm(z0.size(0), device=z0.device)[:count])
    anchor_ref = F.normalize(z0[anchor_idx], dim=-1).detach()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.mtg_lr, weight_decay=1e-5)
    with torch.no_grad():
        logits = head(z0[support])
        initial_ce = F.cross_entropy(logits, targets).item()
    trace = [dict(stage='initial', step=0, support_ce=initial_ce,
                  support_acc=(logits.argmax(-1) == targets).float().mean().item(),
                  anchor_loss=0., objective=initial_ce, adapter_grad_norm=0.)]
    last_grad = 0.
    for step in range(args.mtg_steps):
        optimizer.zero_grad(set_to_none=True)
        z = encode(model, data, args, dad)
        logits = head(z[support])
        ce = F.cross_entropy(logits, targets)
        anchor = (F.normalize(z[anchor_idx], dim=-1) - anchor_ref).square().sum(-1).mean()
        loss = ce + args.mtg_anchor * anchor
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite MTG loss')
        loss.backward()
        last_grad = torch.stack([p.grad.detach().square().sum() for p in params
                                 if p.grad is not None]).sum().sqrt().item()
        if not math.isfinite(last_grad):
            raise RuntimeError('Nonfinite MTG gradients')
        trace.append(dict(stage='anchored', step=step, support_ce=ce.item(),
                          support_acc=(logits.argmax(-1) == targets).float().mean().item(),
                          anchor_loss=anchor.item(), objective=loss.item(), adapter_grad_norm=last_grad))
        optimizer.step()
    with torch.no_grad():
        z = encode(model, data, args, dad)
        logits = head(z[support])
        predictions = torch.cat([classes[head(z[chunk]).argmax(-1)] for chunk in query.split(500)])
        diagnostics = dict(
            support_ce=F.cross_entropy(logits, targets).item(),
            support_acc=(logits.argmax(-1) == targets).float().mean().item(),
            representation_drift=(F.normalize(z, dim=-1) - F.normalize(z0, dim=-1)).square().sum(-1).mean().item(),
            last_adapter_grad_norm=last_grad, anchor_nodes=anchor_idx.numel(),
            adapter_parameters=sum(p.numel() for p in params))
    return predictions, diagnostics, trace


def evaluate_mtg(backbone, data, args):
    for name in ('mtg_lr', 'mtg_anchor'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be finite and positive')
    if min(args.mtg_steps, args.mtg_anchor_nodes, args.DAD) < 0 or args.num_trials < 1:
        raise ValueError('Invalid steps, anchor count, unresolved DAD or trial count')
    output = Path(args.mtg_output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(dict(vars(args), project='LoGo', method_name='anchored',
        adapter_vectors=2, adapter_temperature=.2, adapter_decay=1e-5,
        pretrain_decay=5e-4, cfn_tau=1.), indent=2), encoding='utf-8')
    backbone.eval()
    device = next(backbone.parameters()).device
    dad = sparse_dad(data.edge_index, data.x1.size(0), data.x1.dtype) if args.DAD else None
    with torch.no_grad():
        z0 = encode(backbone, data, args, dad).detach()
    labels = data.y.reshape(-1)
    accuracies = []
    print(f'[LoGo anchored] output={output.resolve()}')
    with (output / 'trials.csv').open('w', newline='', encoding='utf-8') as results, \
         (output / 'training.csv').open('w', newline='', encoding='utf-8') as curves:
        writer = curve_writer = None
        for trial in range(1, args.num_trials + 1):
            folder = Path(args.sample_root) / args.test_dataset / f'{args.shot_num}_shot' / str(trial)
            support, query = [torch.load(folder / name, map_location='cpu', weights_only=True).long().reshape(-1).to(device)
                              for name in ('train_idx.pt', 'test_idx.pt')]
            for idx in (support, query):
                if not idx.numel() or idx.min() < 0 or idx.max() >= z0.size(0) or idx.unique().numel() != idx.numel():
                    raise ValueError(f'Trial {trial}: invalid or duplicate indices')
            if torch.isin(support, query).any():
                raise ValueError('Support/query overlap')
            devices = [device.index] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(args.seed + trial)
                if devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(args.seed + trial)
                predictions, diagnostics, trace = fit_mtg_trial(
                    backbone, data, z0, support, labels[support], query, args, dad)
            accuracy = (predictions == labels[query]).float().mean().item()
            row = dict(trial=trial, **diagnostics, anchored_acc=accuracy)
            if writer is None:
                writer = csv.DictWriter(results, fieldnames=list(row))
                writer.writeheader()
                curve_writer = csv.DictWriter(curves, fieldnames=['trial'] + list(trace[0]))
                curve_writer.writeheader()
            writer.writerow(row)
            curve_writer.writerows(dict(trial=trial, **item) for item in trace)
            results.flush()
            curves.flush()
            accuracies.append(accuracy)
            if args.verbose:
                print(f'[LoGo anchored] trial={trial}/{args.num_trials} ACC={accuracy:.4f}')
    values = torch.tensor(accuracies, dtype=torch.float64)
    summary = dict(anchored_acc=dict(mean=values.mean().item(), std=values.std(unbiased=False).item()))
    print(f"[LoGo anchored] ACC: {summary['anchored_acc']['mean']:.4f} +/- {summary['anchored_acc']['std']:.4f}")
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return accuracies
