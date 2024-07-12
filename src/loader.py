import clip
import torch
import torch.nn as nn
from pathlib import Path
from .utilities import temporarily_add_to_path
from .mapper import LatentMapper
from arcface.models import resnet_face18

class Loader:
    def __init__(self, device) -> None:
        self.device = device

    def load_stylegan(self):
        model_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
        with temporarily_add_to_path("./stylegan3"):
            from stylegan3 import legacy, dnnlib
            with dnnlib.util.open_url(model_url) as f:
                G = legacy.load_network_pkl(f)['G_ema']
                G.eval()
                G = G.to(self.device)
            return G

    
    def load_clip(self):
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
        clip_model.eval()
        clip_model = clip_model.to(self.device)
        return clip_model


    def load_arcface(self, path: str = "arcface/arcface.pth"):
        arcface_model = resnet_face18(False).to(self.device)
        out = nn.DataParallel(arcface_model)
        pretrained_dict = torch.load(path,map_location=torch.device('cpu'))
        out.load_state_dict(pretrained_dict, strict=False)
        arcface_model = out.module.to(self.device)
        arcface_model.eval()
        return arcface_model

    def load_mapper(self,latent_dim=512, num_layers_in_group = 4, edit_coarse = True, edit_medium = True, edit_fine = True):
        latent_mapper = LatentMapper(latent_dim,num_layers_in_group, edit_coarse,edit_medium, edit_fine).to(self.device)
        return latent_mapper

    def load_pretrained_mapper(self, path: str):
        mapper = torch.load(path)
        mapper.eval()
        mapper = mapper.to(self.device)
        return mapper

    def load_encoder(self, encoder_path: str = "./editing/restyle_pSp_ffhq.pt"):
        with temporarily_add_to_path("./editing"):
            from utils.inference_utils import load_encoder
            encoder, opts = load_encoder(checkpoint_path=encoder_path)
            encoder.eval()
            opts.resize_outputs = False
            opts.n_iters_per_batch = 5
            return encoder, opts
    
        

    