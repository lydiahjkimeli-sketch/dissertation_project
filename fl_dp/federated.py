from dataclasses import dataclass
from typing import List, Dict
import numpy as np, torch, torch.nn as nn
from .model import MLP
from .dp_sgd import per_sample_gradients, clip_and_aggregate, add_gaussian_noise, apply_gradients
@dataclass
class ClientConfig:
    lr: float = 1e-3; local_epochs: int = 3; batch_size: int = 64; clip_norm: float = 1.0; noise_multiplier: float = 0.8
class Client:
    def __init__(self, cid:int, x, y, in_dim:int, device='cpu', cfg:ClientConfig=ClientConfig()):
        self.cid=cid; self.x=torch.tensor(x,dtype=torch.float32,device=device); self.y=torch.tensor(y,dtype=torch.float32,device=device)
        self.model=MLP(in_dim).to(device); self.cfg=cfg; self.loss_fn=nn.BCEWithLogitsLoss(); self.device=device
    def set_weights(self, state): self.model.load_state_dict(state)
    def get_weights(self): return {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
    def local_train(self):
        N=self.x.size(0); idx=np.arange(N)
        for _ in range(self.cfg.local_epochs):
            np.random.shuffle(idx)
            for s in range(0,N,self.cfg.batch_size):
                b=idx[s:s+self.cfg.batch_size]; xb=self.x[b]; yb=self.y[b]
                psg=per_sample_gradients(self.model,self.loss_fn,xb,yb)
                clipped=clip_and_aggregate(psg,self.cfg.clip_norm)
                noisy=add_gaussian_noise(clipped,self.cfg.noise_multiplier,self.cfg.clip_norm,xb.device)
                apply_gradients(self.model,noisy,self.cfg.lr)
        return self.get_weights()
def fedavg(state_dicts: List[Dict[str, torch.Tensor]]):
    avg={}
    for k in state_dicts[0].keys(): avg[k]=sum(sd[k] for sd in state_dicts)/len(state_dicts)
    return avg
def evaluate(model_state, x, y, in_dim, device='cpu'):
    model=MLP(in_dim).to(device); model.load_state_dict(model_state); model.eval()
    with torch.no_grad(): logits=model(torch.tensor(x,dtype=torch.float32,device=device)).cpu().numpy()
    return 1/(1+np.exp(-logits))
