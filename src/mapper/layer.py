import math
import torch
import torch.nn as nn
from torch.nn import functional as F 

class MapperLayer(nn.Module):
    def __init__(
        self, dim, bias_init=0, lr_mul=0.01
    ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(dim, dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(dim).fill_(bias_init))
        self.scale = (1 / math.sqrt(dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        # out = F.leaky_relu(out)
        out = fused_leaky_relu(out, self.bias * self.lr_mul)
        return out

def fused_leaky_relu(input: torch.Tensor, bias: torch.Tensor, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    input = input.cuda()
    return (
        F.leaky_relu(input + bias.view(1, *rest_dim, bias.shape[0]), negative_slope=negative_slope) * scale
    )