import argparse
import os
import platform
import re
import warnings
from typing import Optional, Union, List

import torch
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
import random
from omegaconf import OmegaConf

from MotionDirector_train import export_to_video, handle_memory_attention, load_primary_models, unet_and_text_g_c, freeze_models
from utils.lora_handler import LoraHandler
from utils.ddim_utils import ddim_inversion
import imageio

from PIL import Image
from models.pipeline_I2V_NoiseRect_zeroscope import NoiseRectSDPipeline
import inspect

def initialize_pipeline(config):
    device = config.device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(config.model_path)

    if config.spatial_lora_path != "" or config.temporal_lora_path != "":
        # Freeze any necessary models
        freeze_models([vae, text_encoder, unet])

        # Enable xformers if available
        handle_memory_attention(config.xformers, config.sdp, unet)
        
        if config.spatial_lora_path != "":
            lora_manager_spatial = LoraHandler(
                version="cloneofsimo",
                use_unet_lora=True,
                use_text_lora=False,
                save_for_webui=False,
                only_for_webui=False,
                unet_replace_modules=["Transformer2DModel"],
                text_encoder_replace_modules=None,
                lora_bias=None
            )
            unet_lora_params, unet_negation = lora_manager_spatial.add_lora_to_model(
                True, unet, lora_manager_spatial.unet_replace_modules, 0, config.spatial_lora_path, r=config.lora_rank, scale=config.spatial_lora_scale)

        if config.temporal_lora_path != "":
            lora_manager_temporal = LoraHandler(
                version="cloneofsimo",
                use_unet_lora=True,
                use_text_lora=False,
                save_for_webui=False,
                only_for_webui=False,
                unet_replace_modules=["TransformerTemporalModel"],
                text_encoder_replace_modules=None,
                lora_bias=None
            )
            unet_lora_params, unet_negation = lora_manager_temporal.add_lora_to_model(
                True, unet, lora_manager_temporal.unet_replace_modules, 0, config.temporal_lora_path, r=config.lora_rank, scale=config.temporal_lora_scale)

    unet.eval()
    text_encoder.eval()
    unet_and_text_g_c(unet, text_encoder, False, False)

    if config.noise_rectification_flag:
        pipe = NoiseRectSDPipeline.from_pretrained(
            pretrained_model_name_or_path=config.model_path,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder.to(device=device, dtype=torch.half),
            vae=vae.to(device=device, dtype=torch.half),
            unet=unet.to(device=device, dtype=torch.half),
        )
    else:
        pipe = TextToVideoSDPipeline.from_pretrained(
            pretrained_model_name_or_path=config.model_path,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder.to(device=device, dtype=torch.half),
            vae=vae.to(device=device, dtype=torch.half),
            unet=unet.to(device=device, dtype=torch.half),
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"[INFO] successfully build pipe ; pipe_name =", pipe.__class__.__name__)
    
    # # Get all function definitions within MyClass
    # functions = inspect.getmembers(pipe.__class__, predicate=inspect.isfunction)

    # # Print the function names
    # for name, func in functions:
    #     print(name)

    return pipe, pipe.__class__.__name__


def inverse_video(pipe, latents, num_steps):
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)

    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt="")[-1]
    return ddim_inv_latent


def prepare_input_latents(
    pipe: Union[TextToVideoSDPipeline 
                ,NoiseRectSDPipeline],
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    latents_path:str,
    noise_prior: float
):
    # initialize with random gaussian noise
    scale = pipe.vae_scale_factor
    shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
    if noise_prior > 0.:
        cached_latents = torch.load(latents_path)
        if 'inversion_noise' not in cached_latents:
            latents = inverse_video(pipe, cached_latents['latents'].unsqueeze(0), 50).squeeze(0)
        else:
            latents = torch.load(latents_path)['inversion_noise'].unsqueeze(0)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)
        if latents.shape != shape:
            latents = interpolate(rearrange(latents, "b c f h w -> (b f) c h w", b=batch_size), (height // scale, width // scale), mode='bilinear')
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
        noise = torch.randn_like(latents, dtype=torch.half)
        latents = (noise_prior) ** 0.5 * latents + (1 - noise_prior) ** 0.5 * noise
    else:
        latents = torch.randn(shape, dtype=torch.half)

    return latents


def encode(pipe: Union[TextToVideoSDPipeline, NoiseRectSDPipeline]
           , pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents


@torch.inference_mode()
def inference(config):
    # convert type from listconfig to list
    prompt = OmegaConf.to_container(config.prompt, resolve=True)
    negative_prompt = OmegaConf.to_container(config.negative_prompt, resolve=True)

    with torch.autocast(config.device, dtype=torch.half):
        config.noise_rectification_flag = (config.input_image != None)
        # prepare models
        pipe, pipe_type = initialize_pipeline(config)
        
        for i in range(config.repeat_num):
            if config.seed is None:
                random_seed = random.randint(100, 10000000)
                torch.manual_seed(random_seed)
            else:
                random_seed = config.seed
                torch.manual_seed(config.seed)

            # prepare input latents
            init_latents = prepare_input_latents(
                pipe=pipe,
                batch_size=len(prompt),
                num_frames=config.num_frames,
                height=config.height,
                width=config.width,
                latents_path=config.latents_path,
                noise_prior=config.noise_prior
            )
            print('init_latents.shape =', init_latents.shape)
            
            with torch.no_grad():
                if pipe_type == 'NoiseRectSDPipeline':
                    video_frames = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=config.width,
                        height=config.height,
                        num_frames=config.num_frames,
                        num_inference_steps=config.num_steps,
                        guidance_scale=config.guidance_scale,
                        latents=init_latents,
                        input_image=config.input_image, # noise rect
                        noise_rectification_period=config.noise_rectification_period,
                        noise_rectification_weight=config.noise_rectification_weight,
                        noise_rectification_weight_start_omega=config.noise_rectification_weight_start_omega,
                        noise_rectification_weight_end_omega=config.noise_rectification_weight_end_omega,
                    ).frames
                else:
                    video_frames = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=config.width,
                        height=config.height,
                        num_frames=config.num_frames,
                        num_inference_steps=config.num_steps,
                        guidance_scale=config.guidance_scale,
                        latents=init_latents
                    ).frames
            
            # =========================================
            # ========= write outputs to file =========
            # =========================================
            os.makedirs(f"{config.output_dir}", exist_ok=True)

            # save to mp4
            export_to_video(video_frames, f"{config.out_name}_{random_seed}.mp4", config.fps)
            # export_to_video(video_frames, f"{config.out_name}.mp4", config.fps)

            # # save to gif
            file_name = f"{config.out_name}_{random_seed}.gif"
            # file_name = f"{out_name}.gif"
            imageio.mimsave(file_name, video_frames, 'GIF', duration=1000 * 1 / config.fps, loop=0)

    return video_frames
    

if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ddim_inversion.yaml")
    parser.add_argument("--xformers", action="store_true", required=False, help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("--sdp", action="store_true", required=False, help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to save output video to")
    parser.add_argument("--output_file_name", type=str, required=False, help="File name of output video")
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    # fmt: on

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    if args.output_dir != None:
        config.output_dir = args.output_dir
    if args.output_file_name != None:
        config.output_file_name = args.output_file_name
    if args.xformers != None:
        config.xformers = args.xformers
    if args.sdp != None:
        config.sdp = args.sdp 

    config.out_name = f"{config.output_dir}/"
    # prompt = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", config.prompt) if platform.system() == "Windows" else config.prompt

    if config.output_file_name != "":
        config.out_name += f"{config.prompt}".replace(' ','_').replace(',', '').replace('.', '')
    else:
        config.out_name += f"{config.output_file_name}"

    config.prompt = [config.prompt] * config.batch_size
    config.negative_prompt = [config.negative_prompt] * config.batch_size


    if config.spatial_lora_path != "":
        print(f"[State] use spatial lora ; lora_path =", config.spatial_lora_path)
        assert os.path.exists(config.spatial_lora_path)
    else:
        print(f"[State] no use spatial lora")
    if config.temporal_lora_path != "":
        print(f"[State] use temporal lora ; lora_path =", config.temporal_lora_path)
        assert os.path.exists(config.temporal_lora_path)
    else:
        print(f"[State] no use temporal lora")
    

    if config.noise_prior > 0:
        # TODO : why random choice?
        # latents_folder = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(config.temporal_path_folder))))}/cached_latents"
        # latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        
        # YP : get the parent folder of the folder whose name contains "checkpoint"
        temp_path = config.temporal_path_folder
        parts = temp_path.split("/")
        checkpoint_index = max([i for i, part in enumerate(parts) if "checkpoint" in part])
        latents_folder = os.path.join("/".join(parts[:checkpoint_index]), 'cached_latents')
        config.latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        assert os.path.exists(config.latents_path)
    else:
        config.latents_path = ""
        
    ## do noise rectification ##
    if config.input_image_path != "":
        assert os.path.exists(config.input_image_path)
        image_type = config.input_image_path.split('.')[-1]
        config.input_image = Image.open(config.input_image_path)
        print(config.input_image.format, config.input_image.size, config.input_image.mode)
        # config.height, config.width = 304, 512
        assert isinstance(config.input_image, Image.Image)
        
        if config.noise_rectification_period == []:
            config.noise_rectification_period = torch.tensor([0, 0.6])
    else:
        config.input_image = None

    # =========================================
    # ============= sample videos =============
    # =========================================

    video_frames = inference(config)


