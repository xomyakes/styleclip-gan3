import torch
import torch.nn as nn
from .layers_group import LayersGroup


class LatentMapper(nn.Module):
    def __init__(self, latent_dim=512, layers_in_group=4, edit_coarse = True, edit_medium = True, edit_fine = True) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.edit_coarse = edit_coarse
        self.edit_medium = edit_medium
        self.edit_fine = edit_fine

        if self.edit_coarse:
            self.coarse_mapper = LayersGroup(latent_dim,layers_in_group)
        if self.edit_medium:
            self.medium_mapper = LayersGroup(latent_dim,layers_in_group)
        if self.edit_fine:
            self.fine_mapper = LayersGroup(latent_dim, layers_in_group)

        
    
    def forward(self, x):
        x_coarse = x[:, :5, :]
        x_medium = x[:, 5:8, :]
        x_fine = x[:, 8:, :]
        
        if self.edit_coarse:
            x_coarse = self.coarse_mapper(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if self.edit_medium:
            x_medium = self.medium_mapper(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if self.edit_fine:
            x_fine = self.fine_mapper(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out