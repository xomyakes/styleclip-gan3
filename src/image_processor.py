

import asyncio
import clip
import gc
import janus
import logging
import os
import torch
from queue import Empty
from torchvision.utils import save_image
import threading
import socketio
from .loader import Loader
from .encoder import prepare_image_to_inversion, get_latents
from .losses import clip_loss, l2_loss, id_loss

logger = logging.getLogger(__name__)

class ImageProcessor:

    def __init__(self, sio) -> None:
        self.sio: socketio.AsyncServer = sio

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loader = Loader(self.device)
        self.stylegan = self.loader.load_stylegan()
        self.clip_model = self.loader.load_clip()
        self.arcface_model = self.loader.load_arcface()
        self.encoder, self.encoder_opts = self.loader.load_encoder()

    async def start(self):
        self.queue = janus.Queue(100)
        self.stop_flag = threading.Event()
        self.processing_thread = threading.Thread(target=self.start_processing,args=[self.queue, self.stop_flag])
        self.processing_thread.start()

    async def stop(self):
        self.stop_flag.set()
        self.processing_thread.join()

    def get_image_paths(self,file_location: str):
        styled_path = os.path.join("./images/styled", os.path.basename(file_location))
        restored_path = os.path.join("./images/restored", os.path.basename(file_location))
        return styled_path, restored_path


    async def add_image_to_queue(self, sid, file_location, prompt, mapper):
        await self.queue.async_q.put((sid, file_location, prompt, mapper))
        return self.get_image_paths(file_location)

    def start_processing(self,queue: janus.Queue, stop_flag: threading.Event):
        asyncio.run(self.process_queue(queue.sync_q, stop_flag))
        
    async def process_queue(self, queue: janus.SyncQueue, stop_flag: threading.Event):
        while not stop_flag.is_set():
            try:
                sid, file_location, prompt, mapper = queue.get(timeout=1)
                styled_path, restored_path =  await self.process_image(file_location, prompt, mapper, sid)
                logger.info(f"Emitting event to {sid=} ")
                await self.sio.emit('image_edited', {'image_path': styled_path, "restored_path" : restored_path},to=sid)
                self.queue.sync_q.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.exception(f"Error processing image: {e}")
                await self.sio.emit('image_edit_error', {'error': str(e)}, room=sid)
                self.queue.sync_q.task_done()
                


    async def process_image(self, file_location :str, prompt: str, mapper: str, sid: str = None):
        _, image_to_restore, average_image, landmarks_transform = prepare_image_to_inversion(os.path.basename(file_location), self.encoder)
        _, result_latents = get_latents(self.encoder, self.encoder_opts, image_to_restore, average_image, landmarks_transform)
        latent = torch.from_numpy(result_latents[0][-1]).unsqueeze(0)
        restored_image = ((self.stylegan.synthesis(latent.to(self.device)) + 1) * 0.5).clamp(0,1).cpu()
        torch.cuda.empty_cache()
        restored_path = os.path.join("./images/restored", os.path.basename(file_location))
        save_image(restored_image,restored_path)
        styled_path = os.path.join("./images/styled",os.path.basename(file_location))
        if prompt:
            await self.optimize_latent(image_to_restore, latent, prompt, styled_path, sid)
        elif mapper:
            await self.apply_mapper(latent, mapper, styled_path)
        torch.cuda.empty_cache()
        return styled_path, restored_path


    async def optimize_latent(self, initial_img: torch.Tensor, latent: torch.Tensor, prompt: str,  save_path: str, sid: str = None):
        num_steps = 100
        lr=0.1
        lambda_l2, lambda_id = 0.008,0.005
        text_features = torch.cat([clip.tokenize(prompt)])
        text_features = self.clip_model.encode_text(text_features.to(self.device)).to(self.device)
        
        w_opt: torch.Tensor = latent.clone().to(self.device)
        w_opt.requires_grad = True
        optimizer = torch.optim.AdamW([w_opt],lr,amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,50,0.5)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            img = ((self.stylegan.synthesis(w_opt) + 1) * 0.5).clamp(0,1).cpu()
            loss_clip = clip_loss(img.to(self.device),text_features, self.clip_model,self.device)
            loss_l2 = l2_loss(latent.to(self.device), w_opt)
            loss_id = id_loss(img.to(self.device), initial_img,self.arcface_model, self.device)
            loss = loss_clip + lambda_l2 * loss_l2 + lambda_id * loss_id
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            if (step + 1) % 10 == 0:
                logger.info(f"Processing {os.path.basename(save_path)} step: {step + 1}/{num_steps}")
                if sid:
                    await self.sio.emit("image_processing",{"step" : step + 1, "steps" : num_steps}, to=sid)
        save_image(img, save_path)

        del optimizer
        del scheduler
        del w_opt
        del text_features
        gc.collect()
        torch.cuda.empty_cache()

    async def apply_mapper(self, latent: torch.Tensor, mapper_name: str, save_path: str):
        mapper_path = os.path.join("./mappers",f"{mapper_name}.pt")
        mapper = self.loader.load_pretrained_mapper(mapper_path)
        mapper.eval()
        delta = mapper(latent.to(self.device)).cpu()
        styled_latent = latent + delta
        styled_image = ((self.stylegan.synthesis(styled_latent.to(self.device)) + 1) / 2).clamp(0,1).cpu()
        save_image(styled_image, save_path)

        del mapper
        del styled_latent
        del delta
        del styled_image
        gc.collect()
        torch.cuda.empty_cache()
        


