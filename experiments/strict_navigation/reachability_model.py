"""TRM-inspired temporal pair head. Added module; official LeWM remains unchanged."""
import torch
from torch import nn
class TemporalHead(nn.Module):
 def __init__(self,dim=192):
  super().__init__();self.net=nn.Sequential(nn.Linear(4*dim,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1),nn.Softplus())
 def forward(self,a,b):
  delta=a-b;return self.net(torch.cat([a,b,delta,delta.abs()],-1)).squeeze(-1)
