from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, TextToVideoSDPipeline
from models.unet_3d_condition import UNet3DConditionModel
from torchvision.io import read_video, write_video
# suppress partial model loading warning
# transformers_logging.set_verbosity_error()

import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision.io import write_video
from pathlib import Path
from omegaconf import OmegaConf
# from util import *
import torchvision.transforms as T
from PIL import Image
import random
import yaml
from einops import rearrange

# from .MotionDirector_inference_multi import initialize_pipeline

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, "w") as file:
        yaml.dump(data, file)
        
def save_video_as_frames(video_path, frames_path, img_size=(512,512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    os.makedirs(frames_path, exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size), resample=Image.Resampling.LANCZOS)
        image_resized.save(f'{frames_path}/{ind}.png')


def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start

def initialize_pipeline(
    pretrained_model_path: str,
    model: str,
    device: str = "cuda",
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)
        noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    unet.eval()
    text_encoder.eval()
    unet._set_gradient_checkpointing(value=False)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=False)

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"[INFO] successfully build pipe ; pipe_name =", pipe.__class__.__name__)

    return pipe



class InversionPipeline(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        model_path = config.pretrained_model_path
        
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision="fp16", torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", revision="fp16", torch_dtype=torch.float16).to(self.device)
        self.unet = UNet3DConditionModel.from_pretrained(model_path, subfolder="unet", revision="fp16", torch_dtype=torch.float16).to(self.device)
        
        # get n_frame_to_invert frames
        self.paths, self.frames = self.load_video_frames(config.frames_path, config.n_frame_to_invert)
        
        self.latents = self.vae.encode(self.frames).latent_dist.sample().mul_(0.18215)
        self.latents = rearrange(self.latents, "(b f) c h w -> b c f h w", b=1).contiguous()
        print(f"self.latents shape: {self.latents.shape}")
        
        self.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        # if config.enable_xformers_memory_efficient_attention:
        #     from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
        #     unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            
        logger.info(f"loaded model from {model_path}")
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        decoded = []
        batch_size = 8
        for b in range(0, latents.shape[0], batch_size):
            latents_batch = 1 / 0.18215 * latents[b:b + batch_size]
            batch_size, channels, num_frames, height, width = latents.shape
            latents_batch = latents_batch.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
            imgs = self.vae.decode(latents_batch).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
            imgs[None, :].reshape((batch_size, num_frames, -1) + imgs.shape[2:]).permute(0, 2, 1, 3, 4)
            decoded.append(imgs)
        return torch.cat(decoded)
    '''
    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=10, deterministic=True):
        print('encode_imgs, imgs.shape =', imgs.shape)
        imgs = 2 * imgs - 1 # normalize first?
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents
    '''
    @torch.no_grad()
    def load_video_frames(self, frames_path, n_frames):
        # load frames
        paths = [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
        if not os.path.exists(paths[0]):
            paths = [f"{frames_path}/%05d.jpg" % i for i in range(n_frames)]
        self.paths = paths
        
        frames = [Image.open(path).convert('RGB') for path in paths]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device) #[f,c,h,w]
        frames = 2 * frames - 1 # normalize
        print('load_video_frames, frames.shape =', frames.shape)

        return paths, frames

    @torch.no_grad()
    def ddim_inversion(self, cond, latent_frames, save_path, batch_size, save_latents=True, timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_to_save = timesteps_to_save if timesteps_to_save is not None else timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, latent_frames.shape[0], batch_size):
                x_batch = latent_frames[b:b + batch_size]
                model_input = x_batch
                # print("model_input.shape in ddim_inversion :", model_input.shape) #[1,4,16,32,48]
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(model_input, t, encoder_hidden_states=cond_batch)["sample"]
                pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
                latent_frames[b:b + batch_size] = mu * pred_x0 + sigma * eps
                
            # print(latent_frames.shape)

            if save_latents and t in timesteps_to_save:
                torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        return latent_frames

    @torch.no_grad()
    def ddim_sample(self, x, cond, batch_size):
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b:b + batch_size]
                model_input = x_batch
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(model_input, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x_batch - sigma * eps) / mu
                x[b:b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
        return x

    @torch.no_grad()
    def extract_ddim_latents(self, num_steps, save_path, batch_size, timesteps_to_save, inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)
        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        latent_frames = self.latents

        inverted_x = self.ddim_inversion(cond, latent_frames, save_path, batch_size=batch_size, save_latents=True, timesteps_to_save=timesteps_to_save)
        print('inverted_x.shape =', inverted_x.shape) #[1,4,16,32,48]
        
        latent_reconstruction = self.ddim_sample(inverted_x, cond, batch_size=batch_size)
        print('latent_reconstruction.shape =', latent_reconstruction.shape) #[1,4,16,32,48]  
                                         
        rgb_reconstruction = self.decode_latents(latent_reconstruction)
        print('rgb_reconstruction.shape =', rgb_reconstruction.shape)
        return rgb_reconstruction


def main(config, device):
    # set a scheduler to generate timesteps to save
    toy_scheduler = DDIMScheduler.from_config(config.pretrained_model_path, subfolder="scheduler",
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
    )
    toy_scheduler.set_timesteps(config.n_save_steps)
    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=config.n_save_steps, strength=1.0)

    seed_everything(1)

    save_path = os.path.join(config.output_dir,
                             f'{config.model_name}',
                             Path(config.src_video_path).stem,
                             f'steps_{config.n_steps}',
                             f'nframes_{config.n_frame_to_invert}')
    logger.info(f"save_path: {save_path}")
    os.makedirs(os.path.join(save_path, f'latents'), exist_ok=True)
    
    # Save inversion prompt in a yaml file
    add_dict_to_yaml_file(os.path.join(config.output_dir, 'inversion_prompts.yaml'), Path(config.src_video_path).stem, config.inversion_prompt)    
    
    # Save config in a yaml file
    with open(os.path.join(save_path, "config.yaml"), "w") as file:
        yaml.dump(OmegaConf.to_container(config), file)
        
    # Main pipeline
    model = InversionPipeline(config, device)
    recon_frames = model.extract_ddim_latents(
                                         num_steps=config.n_steps,
                                         save_path=save_path,
                                         batch_size=config.batch_size,
                                         timesteps_to_save=timesteps_to_save,
                                         inversion_prompt=config.inversion_prompt,
    )

    # pipe = initialize_pipeline(args.model, args.pretrained_model_path, device) # without any lora, use TextToVideoSDPipeline
    # ddim_inversion(pipe, toy_scheduler, , config.n_save_step, config.inversion_prompt)


    if not os.path.isdir(os.path.join(save_path, f'frames')):
        os.mkdir(os.path.join(save_path, f'frames'))
    for i, frame in enumerate(recon_frames):
        T.ToPILImage()(frame).save(os.path.join(save_path, f'frames', f'{i:05d}.png'))
    frames = (recon_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(os.path.join(save_path, f'inverted.mp4'), frames, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ddim_inversion.yaml")
    parser.add_argument("--video_path", type=str, required=False, help="Path to the video to invert.")
    parser.add_argument("--gpu", type=int, required=False, help="GPU number to use.")
    parser.add_argument("--width", type=int, required=False, help="")
    parser.add_argument("--height", type=int, required=False, help="")

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    # Overwrite config with command line arguments
    if args.video_path is not None:
        config.src_video_path = args.video_path
    if args.gpu is not None:
        self.device = f"cuda:{args.gpu}"
    if args.width is not None and args.height is not None:
        config.image_size = [args.height, args.width]

    
    # Set up logging
    transformers_logging.set_verbosity_error()
    logging_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"config: {config}")

    # TODO: better to avoid save then load
    assert os.path.exists(config.src_video_path), f"src_video_path {config.src_video_path} does not exist."
    save_video_as_frames(config.src_video_path, config.frames_path, img_size=(config.image_size[1], config.image_size[0]))  # (w, h)
    config.src_video_path = os.path.join(Path(config.src_video_path).parent, Path(config.src_video_path).stem)

    device = torch.device(config.device)
    seed_everything(config.seed)
    main(config, device)