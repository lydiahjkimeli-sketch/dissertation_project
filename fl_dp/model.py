import torch, torch.nn as nn
class MLP(nn.Module):
    def __init__(self, in_dim:int, hidden:int=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)
