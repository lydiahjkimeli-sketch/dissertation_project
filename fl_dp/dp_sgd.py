import torch, torch.nn as nn
from typing import Iterable
def per_sample_gradients(model: nn.Module, loss_fn, x, y):
    grads = []
    for i in range(x.size(0)):
        model.zero_grad(set_to_none=True)
        loss_fn(model(x[i:i+1]), y[i:i+1]).backward()
        grads.append([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()])
    return grads
def clip_and_aggregate(per_sample_grads: Iterable, clip_norm: float):
    batch = len(per_sample_grads)
    clipped = []
    for g_list in per_sample_grads:
        total = torch.sqrt(sum((g.pow(2).sum() for g in g_list)))
        scale = min(1.0, clip_norm / (total + 1e-12))
        clipped.append([g*scale for g in g_list])
    avg = []
    for params in zip(*clipped): avg.append(sum(params)/batch)
    return avg
def add_gaussian_noise(grad_list, noise_multiplier: float, clip_norm: float, device):
    if noise_multiplier<=0: return grad_list
    return [g + torch.randn_like(g, device=device)*noise_multiplier*clip_norm for g in grad_list]
def apply_gradients(model: nn.Module, grads: Iterable, lr: float):
    with torch.no_grad():
        for p,g in zip(model.parameters(), grads): p.add_(-lr*g)
