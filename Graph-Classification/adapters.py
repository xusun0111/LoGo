import copy
import torch
from torch import nn
from torch.nn import functional as F

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

def cosine_logits(embeddings, centers, temperature=0.2):
    return F.normalize(embeddings, dim=-1) @ F.normalize(centers, dim=-1).T / temperature
