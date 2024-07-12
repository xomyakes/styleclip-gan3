import torch
import torch.nn as nn
from torch.nn import functional as F

from .layer import MapperLayer

class LayersGroup(nn.Module):
    def __init__(self, latent_dim=512, num_layers = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        # self.pixel_norm = PixelNorm()
        # self.layers = []
        # for _ in range(self.num_layers):
        #     fc = nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        #     with torch.no_grad():
        #         fc.weight.fill_(0.)
        #         fc.bias.fill_(0.)
        #     self.layers.append(fc)
            # relu = nn.ReLU(inplace=False)
            # layers.append(relu)
            # layers.append(nn.InstanceNorm1d(self.latent_dim))
            # layers.append(nn.LeakyReLU())
        self.mapper = nn.Sequential(
            
            PixelNorm(),
            # *layers
            *[MapperLayer(latent_dim) for _ in range(self.num_layers)]
        )

    def forward(self, input):
        # x = self.pixel_norm(input)
        # for layer in self.layers:
        #     x = layer(x)
        #     x = F.leaky_relu(x,inplace=False)
        out = self.mapper(input)
        return out
    

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

        