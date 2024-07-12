import torch
import torchvision.transforms as tt
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image



def clip_loss(image: torch.Tensor, text_features: torch.Tensor, clip_model, device):
    clip_transform = tt.Compose([
        tt.Resize(clip_model.visual.input_resolution,interpolation=tt.InterpolationMode.BICUBIC),
        tt.CenterCrop(clip_model.visual.input_resolution),
        tt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image_features = clip_model.encode_image(clip_transform(image[0]).unsqueeze(0).to(device)).to(device)
    loss = 1 - F.cosine_similarity(image_features, text_features).min()
    return loss


def l2_loss(fixed_latent: torch.Tensor, latent: torch.Tensor):
    loss = F.mse_loss(latent, fixed_latent)
    return loss


arcface_transform = tt.Compose([
    tt.Grayscale(num_output_channels=1), # Т.к. arcface работает с черно-белыми изображениями
    tt.Resize((128, 128)),
    tt.ToTensor(),
    tt.Normalize([0.5], [0.5]) 
])

def get_face_embeddings(img: torch.Tensor, arcface_model, device):
    arcface_model.eval()
    img_transformed = arcface_transform(to_pil_image(img[0].detach().cpu())).unsqueeze(0).to(device)
    embeddings = arcface_model(img_transformed)
    return embeddings

def id_loss(image, initial_image, arcface_model, device):
    current_embeddings = get_face_embeddings(image, arcface_model, device)
    initial_embeddings = get_face_embeddings(initial_image, arcface_model, device)
    loss = 1 - F.cosine_similarity(current_embeddings,initial_embeddings)
    return loss