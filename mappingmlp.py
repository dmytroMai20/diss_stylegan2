import torch
from torch import nn
import torch.nn.functional as F
from util import EqualizedLinear

class MappingMLP(nn.Module):
    def __init__(self, features, num_layers):
        super().__init__()
        layers =[]
        for i in range(num_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        z = F.normalize(z, dim=1)
        return self.net(z)