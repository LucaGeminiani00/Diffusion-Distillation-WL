import math
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils import data as D


##Class used when utilizing distillation based on W-GAN
class Critic(nn.Module):
    def __init__(
        self,
        n_feat,
        critic_dropout = 0.0,  
        **kwargs
    ):
        super().__init__()
        d_in = [n_feat] + [128, 128, 128]  
        d_out = [128, 128, 128] + [1]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(critic_dropout)
        self.opt = Adam(self.parameters(), lr = 0.001) 

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(F.relu(layer(x)))
        return self.layers[-1](x)     #Current output is [64,24,1]

    def gradient_penalty(self, x, x_hat):
        alpha = torch.rand(x.size(1)).unsqueeze(1).to(x.device)
        interpolated = x * alpha + x_hat * (1 - alpha)
        interpolated = torch.autograd.Variable(interpolated.detach(), requires_grad=True)
        critic = self(interpolated)
        gradients = torch.autograd.grad(critic, interpolated, torch.ones_like(critic),
                                        retain_graph=True, create_graph=True, only_inputs=True)[0]
        penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()             # one-sided
        #penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()          # two-sided 
        return penalty
    
