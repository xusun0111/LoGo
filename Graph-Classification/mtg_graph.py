import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from adapters import adapted_copy, prototypes, cosine_logits


def graph_batches(graphs, args):
    return DataLoader(graphs, batch_size=args.batch_size, shuffle=False)


def encode_batch(model, batch, args):
    device = next(model.parameters()).device
    batch = batch.to(device)
    # Model concatenates encoder outputs and propagates both on original edges.
    nodes = model(batch.x1, batch.edge_index, batch.x2, batch.edge_index2,
                  args, train=0)
    return global_mean_pool(nodes, batch.batch, size=batch.num_graphs)


def encode_graphs(model, graphs, args):
    return torch.cat([encode_batch(model, batch, args)
                      for batch in graph_batches(graphs, args)], dim=0)


def fit_frozen_trial(backbone, graphs, z0, support, support_labels, query, args):
    device = z0.device
    support = torch.as_tensor(support, dtype=torch.long, device=device)
    query = torch.as_tensor(query, dtype=torch.long, device=device)
    labels = support_labels.to(device)
    classes, targets = torch.unique(labels, sorted=True, return_inverse=True)
    if len(classes) < 2:
        raise ValueError('At least two support classes are required')
    centers = prototypes(z0[support], labels, classes).detach()
    head = lambda z: cosine_logits(z, centers)
    support_graphs = [graphs[int(i)] for i in support.cpu()]
    count = args.mtg_anchor_graphs
    anchors = (torch.arange(len(graphs)) if count == 0 or count >= len(graphs)
               else torch.randperm(len(graphs))[:count])
    anchor_graphs = [graphs[int(i)] for i in anchors]
    anchor_ref = F.normalize(z0[anchors.to(device)], dim=1).detach()
    initial = adapted_copy(backbone).eval()
    predictions, info, trace = {}, {}, []

    def report(name, z):
        logits = head(z[support])
        info[name + '_support_ce'] = F.cross_entropy(logits, targets).item()
        info[name + '_support_acc'] = (logits.argmax(1) == targets).float().mean().item()
        info[name + '_representation_drift'] = (
            F.normalize(z, dim=1) - F.normalize(z0, dim=1)
        ).square().sum(1).mean().item()
        predictions[name] = classes[head(z[query]).argmax(1)].cpu()

    with torch.no_grad():
        initial_ce = F.cross_entropy(head(z0[support]), targets).item()
    trace.append(dict(stage='initial', step=0, support_ce=initial_ce,
                      anchor_loss=0., objective=initial_ce, adapter_grad_norm=0.))
    model = initial
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.mtg_lr, weight_decay=1e-5)
    for step in range(args.mtg_steps):
        optimizer.zero_grad(set_to_none=True)
        support_ce, anchor_loss, offset = 0., 0., 0
        for batch in graph_batches(support_graphs, args):
            z = encode_batch(model, batch, args)
            size = z.size(0)
            loss = F.cross_entropy(head(z), targets[offset:offset+size], reduction='sum') / len(support)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite support loss')
            loss.backward()
            support_ce += loss.item()
            offset += size
        if args.mtg_anchor:
            offset = 0
            for batch in graph_batches(anchor_graphs, args):
                z = encode_batch(model, batch, args)
                size = z.size(0)
                drift = (F.normalize(z, dim=1) - anchor_ref[offset:offset+size]).square().sum() / len(anchors)
                if not torch.isfinite(drift):
                    raise RuntimeError('Nonfinite graph anchor loss')
                (args.mtg_anchor * drift).backward()
                anchor_loss += drift.item()
                offset += size
        norm = torch.stack([p.grad.square().sum() for p in params if p.grad is not None]).sum().sqrt().item()
        if not np.isfinite(norm):
            raise RuntimeError('Nonfinite adapter gradients')
        trace.append(dict(stage='anchored', step=step, support_ce=support_ce,
                          anchor_loss=anchor_loss, objective=support_ce + args.mtg_anchor*anchor_loss,
                          adapter_grad_norm=norm))
        optimizer.step()
    with torch.no_grad():
        report('anchored', encode_graphs(model, graphs, args))
    info['anchor_graphs'] = len(anchors)
    info['adapter_parameters'] = sum(p.numel() for p in initial.parameters() if p.requires_grad)
    return predictions, info, trace


def evaluate(backbone, graphs, args):
    backbone.eval()
    labels = np.array([int(g.y.reshape(-1)[0]) for g in graphs])
    classes = np.unique(labels)
    indices = np.arange(len(graphs))
    with torch.no_grad():
        z0 = encode_graphs(backbone, graphs, args).detach()
    output = Path(args.mtg_output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), project='LoGo', method_name='anchored',
                  adapter_vectors=2, adapter_temperature=.2, adapter_decay=1e-5,
                  pretrain_decay=5e-4, cfn_tau=1., support_fraction=.2, query_fraction=.8,
                  anchor_unit='graph', propagation_topology='original_for_both')
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    with (output/'trials.csv').open('w', newline='', encoding='utf-8') as result_file, \
         (output/'training.csv').open('w', newline='', encoding='utf-8') as curve_file, \
         (output/'splits.jsonl').open('w', encoding='utf-8') as split_file:
        writer = curve_writer = None
        for trial in range(1, args.num_trials+1):
            split_seed = args.seed+trial
            support, query = train_test_split(indices, test_size=.8, stratify=labels,
                                              random_state=split_seed)
            if not np.array_equal(np.unique(labels[support]), classes):
                raise ValueError('The 20% support split must contain every class')
            split_file.write(json.dumps(dict(trial=trial, support=support.tolist(), query=query.tolist()))+'\n')
            device = z0.device
            devices = [device.index] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(args.seed+trial)
                if devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(args.seed+trial)
                predictions, info, trace = fit_frozen_trial(
                    backbone, graphs, z0, support, torch.as_tensor(labels[support]), query, args)
            row = dict(trial=trial, **info)
            for name, predicted in predictions.items():
                pred = predicted.numpy()
                row[name+'_acc'] = accuracy_score(labels[query], pred)
                row[name+'_f1'] = f1_score(labels[query], pred, average='macro', zero_division=0)
            if writer is None:
                writer = csv.DictWriter(result_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            result_file.flush()
            if trace:
                if curve_writer is None:
                    curve_writer = csv.DictWriter(curve_file, fieldnames=['trial']+list(trace[0]))
                    curve_writer.writeheader()
                curve_writer.writerows(dict(trial=trial, **item) for item in trace)
                curve_file.flush()
            rows.append(row)
            if args.verbose:
                print(f'[Trial {trial}/{args.num_trials}] anchored_acc={row["anchored_acc"]:.4f}')
    metrics = ('anchored_acc', 'anchored_f1')
    summary = {}
    for name in metrics:
        values = np.array([row[name] for row in rows])
        summary[name] = dict(mean=float(values.mean()), std=float(values.std()))
        print(f'{name}: {values.mean():.4f} +/- {values.std():.4f}')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Results: {output.resolve()}')
    return output, rows
