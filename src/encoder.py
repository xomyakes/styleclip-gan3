import os
import torch
import torchvision.transforms as tt
from torchvision.io import read_image 
import time
from .utilities import temporarily_add_to_path

def prepare_image_to_inversion(image_name, encoder):
    images_folder = "./images"
    image_path = os.path.join(images_folder,"real",image_name)
    aligned_folder = os.path.join(images_folder,"aligned")
    cropped_folder = os.path.join(images_folder,"cropped")
    if not os.path.isdir(aligned_folder):
        os.mkdir(aligned_folder)
    if not os.path.isdir(cropped_folder):
        os.mkdir(cropped_folder)
    real_image= read_image(image_path).unsqueeze(0)
    # Загрузим оригинальное изображение и преобразуем его в RGB
    # original_image = Image.open(image_path).convert('RGB')
    img_transforms = tt.Compose([
            tt.Resize((256, 256)),
            tt.ToTensor(),
            tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    with temporarily_add_to_path("./editing"):
        from notebooks.notebook_utils import run_alignment, crop_image, compute_transforms
        from utils.inference_utils import get_average_image
        # Выполним выравнивание и кроп изображения
        aligned_image = run_alignment(image_path)
        aligned_path = os.path.join(aligned_folder,image_name)
        aligned_image.save(aligned_path)
        cropped_image = crop_image(image_path)
        cropped_path = os.path.join(cropped_folder,image_name)
        cropped_image.save(cropped_path)
        # joined = np.concatenate([aligned_image.resize((256, 256)), cropped_image.resize((256, 256))], axis=1)
        landmarks_transform = compute_transforms(aligned_path=aligned_path, cropped_path=cropped_path)
        average_image = get_average_image(encoder)
        transformed_image = img_transforms(aligned_image)
        return real_image, transformed_image, average_image, landmarks_transform

def get_latents(encoder, opts, image, avg_image,landmarks_transform):
    with temporarily_add_to_path("./editing"):
        from utils.inference_utils import run_on_batch
        with torch.no_grad():
            tic = time.time()
            result_batch, result_latents = run_on_batch(inputs=image.unsqueeze(0).cuda().float(),
                net=encoder,
                opts=opts,
                avg_image=avg_image,
                landmarks_transform=torch.from_numpy(landmarks_transform).cuda().float())
            toc = time.time()
            print('Inference took {:.4f} seconds.'.format(toc - tic))
            print(result_latents[0][0].shape)
            return (result_batch, result_latents)