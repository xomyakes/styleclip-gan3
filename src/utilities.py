import os
import sys
import torch
from torchvision import transforms as tt
from contextlib import contextmanager
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


@contextmanager
def temporarily_add_to_path(path):
    old_path = sys.path.copy()
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = old_path

def show_image(image: torch.Tensor):
    img_to_show = image.clone().detach()
    plt.imshow(img_to_show[0].permute(1, 2, 0).detach().cpu())
    plt.grid(None)
    plt.axis("off")
    plt.show()

def show_loss_plots(losses: Dict[str, List[float]]):
    for key, val in losses.items():
        epochs = list(range(1, len(val) + 1))
        plt.plot(epochs, val, marker='.', linestyle='-', label=f'{key} loss')
    plt.title('Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_image(stylegan, device) -> Tuple[torch.Tensor ,torch.Tensor ,torch.Tensor]:
    z: torch.Tensor = torch.randn(1, stylegan.z_dim)
    with torch.no_grad():
        w: torch.Tensor = stylegan.mapping(z.to(device),None).cpu()
        image: torch.Tensor = ((stylegan.synthesis(w.to(device)) + 1) / 2).clamp(0,1).cpu()
    torch.cuda.empty_cache()
    return image, w, z

def print_memory_usage():
    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()
    print(f"Allocated memory: {allocated_memory / 1e6} MB")
    print(f"Cached memory: {cached_memory / 1e6} MB")

