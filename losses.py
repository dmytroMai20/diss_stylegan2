import torch
from torch import nn
import torch.nn.functional as F

class DiscriminatorLoss(nn.Module):
    def forward(self, f_real: torch.Tensor, f_fake: torch.Tensor):
        loss_real = F.softplus(-f_real).mean()
        loss_fake = F.softplus(f_fake).mean()
        return loss_real, loss_fake


class GeneratorLoss(nn.Module):
    def forward(self, f_fake: torch.Tensor):
        return F.softplus(-f_fake).mean()