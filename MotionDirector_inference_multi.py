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

from MotionDirector_train import export_to_video, handle_memory_attention, load_primary_models, unet_and_text_g_c, freeze_models
from utils.lora_handler import LoraHandler
from utils.ddim_utils import ddim_inversion
import imageio

from PIL import Image
from models.pipeline_I2V_NoiseRect_zeroscope import NoiseRectSDPipeline
import inspect

def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    spatial_lora_path: str = "",
    temporal_lora_path: str = "",
    lora_rank: int = 64,
    spatial_lora_scale: float = 1.0,
    temporal_lora_scale: float = 1.0,
    
    noise_rectification_flag: bool = False, # whether to use noise rectification technique 
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)

    if spatial_lora_path != None or temporal_lora_path != None:
        # Freeze any necessary models
        freeze_models([vae, text_encoder, unet])

        # Enable xformers if available
        handle_memory_attention(xformers, sdp, unet)
        
        if spatial_lora_path != None:
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
                True, unet, lora_manager_spatial.unet_replace_modules, 0, spatial_lora_path, r=lora_rank, scale=spatial_lora_scale)

        if temporal_lora_path != None:
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
                True, unet, lora_manager_temporal.unet_replace_modules, 0, temporal_lora_path, r=lora_rank, scale=temporal_lora_scale)

    unet.eval()
    text_encoder.eval()
    unet_and_text_g_c(unet, text_encoder, False, False)

    if noise_rectification_flag:
        pipe = NoiseRectSDPipeline.from_pretrained(
            pretrained_model_name_or_path=model,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder.to(device=device, dtype=torch.half),
            vae=vae.to(device=device, dtype=torch.half),
            unet=unet.to(device=device, dtype=torch.half),
        )
    else:
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
def inference(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    num_steps: int = 50,
    guidance_scale: float = 15,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    spatial_lora_path: str = "",
    temporal_lora_path: str = "",
    lora_rank: int = 64,
    spatial_lora_scale: float = 1.0,
    temporal_lora_scale: float = 1.0,
    seed: Optional[int] = None,
    latents_path: str="",
    noise_prior: float = 0.,
    repeat_num: int = 1,
    out_name: str = "",
    
    input_image: Optional[Image.Image] = None,
    noise_rectification_period: Optional[list] = None,
    noise_rectification_weight: Optional[Tensor] = None,
    noise_rectification_weight_start_omega: float = 1.0,
    noise_rectification_weight_end_omega: float = 0.5,
):

    with torch.autocast(device, dtype=torch.half):
        noise_rectification_flag = (input_image != None)
        # prepare models
        pipe, pipe_type = initialize_pipeline(model, device, xformers, sdp, spatial_lora_path, temporal_lora_path, lora_rank,
                                   spatial_lora_scale, temporal_lora_scale, noise_rectification_flag)
        
        for i in range(repeat_num):
            if seed is None:
                random_seed = random.randint(100, 10000000)
                torch.manual_seed(random_seed)
            else:
                random_seed = seed
                torch.manual_seed(seed)

            # prepare input latents
            init_latents = prepare_input_latents(
                pipe=pipe,
                batch_size=len(prompt),
                num_frames=num_frames,
                height=height,
                width=width,
                latents_path=latents_path,
                noise_prior=noise_prior
            )
            
            with torch.no_grad():
                if pipe_type == 'NoiseRectSDPipeline':
                    video_frames = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        latents=init_latents,
                        input_image=input_image,
                        noise_rectification_period=noise_rectification_period,
                        noise_rectification_weight=noise_rectification_weight,
                        noise_rectification_weight_start_omega=noise_rectification_weight_start_omega,
                        noise_rectification_weight_end_omega=noise_rectification_weight_end_omega,
                    ).frames
                else:
                    video_frames = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        latents=init_latents
                    ).frames
            
            # =========================================
            # ========= write outputs to file =========
            # =========================================
            os.makedirs(args.output_dir, exist_ok=True)

            # save to mp4
            export_to_video(video_frames, f"{out_name}_{random_seed}.mp4", args.fps)
            # export_to_video(video_frames, f"{out_name}.mp4", args.fps)

            # # save to gif
            file_name = f"{out_name}_{random_seed}.gif"
            # file_name = f"{out_name}.gif"
            imageio.mimsave(file_name, video_frames, 'GIF', duration=1000 * 1 / args.fps, loop=0)

    return video_frames
    

if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Text prompt to condition on")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-o", "--output_dir", type=str, default="./outputs/inference", help="Directory to save output video to")
    parser.add_argument("-ofn", "--output_file_name", type=str, default=None, help="File name of output video")
    parser.add_argument("-B", "--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("-W", "--width", type=int, default=384, help="Width of output video")
    parser.add_argument("-H", "--height", type=int, default=384, help="Height of output video")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-s", "--num-steps", type=int, default=30, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=12, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-f", "--fps", type=int, default=8, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-slp", "--spatial_path_folder", type=str, default=None, help="Path to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-tlp", "--temporal_path_folder", type=str, default=None,
                        help="Path to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-lr", "--lora_rank", type=int, default=32, help="Size of the LoRA checkpoint's projection matrix (defaults to 32).")
    parser.add_argument("-sps", "--spatial_path_scale", type=float, default=1.0, help="Scale of spatial LoRAs.")
    parser.add_argument("-tps", "--temporal_path_scale", type=float, default=1.0, help="Scale of temporal LoRAs.")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    parser.add_argument("-np", "--noise_prior", type=float, default=0., help="Scale of the influence of inversion noise.")
    parser.add_argument("-ci", "--checkpoint_index", type=str, default="default",
                        help="The index of checkpoint, such as 300.")
    parser.add_argument("-rn", "--repeat_num", type=int, default=1,
                        help="How many results to generate with the same prompt.")
    
    # parser.add_argument("-uNR", "--use_noise_rectification", type=bool, default=False,
    #                     help="Whether use noise rectification technique. (defaults to False)")
    parser.add_argument("-img", "--input_image_path", type=str, default=None ,help="Image guidanced for noise rectification technique")
    parser.add_argument("-NR_p", "--noise_rectification_period", type=list, default=None, help="noise rectification time period (start and end point), such as [0, 0.6]")
    parser.add_argument("-NR_w", "--noise_rectification_weight", type=list, default=None, help="the scale on each generated frames")
    parser.add_argument("-NR_sO", "--noise_rectification_weight_start_omega", type=float, default = 1.0, help="start scale value (if weight not be set)")
    parser.add_argument("-NR_eO", "--noise_rectification_weight_end_omega", type=float, default = 0.5, help="end scale value (if weight not be set)")

    args = parser.parse_args()
    # fmt: on

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    out_name = f"{args.output_dir}/"
    prompt = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", args.prompt) if platform.system() == "Windows" else args.prompt
    if args.output_file_name == None:
        out_name += f"{prompt}".replace(' ','_').replace(',', '').replace('.', '')
    else:
        out_name += f"{args.output_file_name}"

    args.prompt = [prompt] * args.batch_size
    if args.negative_prompt is not None:
        args.negative_prompt = [args.negative_prompt] * args.batch_size
        
    if args.spatial_path_folder != None:
        print(f"[State] use spatial lora ; lora_path =", args.spatial_path_folder)
        assert os.path.exists(args.spatial_path_folder)
    else:
        print(f"[State] no use spatial lora")
    if args.temporal_path_folder != None:
        print(f"[State] use temporal lora ; lora_path =", args.temporal_path_folder)
        assert os.path.exists(args.temporal_path_folder)
    else:
        print(f"[State] no use temporal lora")
    

    if args.noise_prior > 0:
        # TODO : why random choice?
        # latents_folder = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.temporal_path_folder))))}/cached_latents"
        # latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        
        # YP : get the parent folder of the folder whose name contains "checkpoint"
        temp_path = args.temporal_path_folder
        parts = temp_path.split("/")
        checkpoint_index = max([i for i, part in enumerate(parts) if "checkpoint" in part])
        latents_folder = os.path.join("/".join(parts[:checkpoint_index]), 'cached_latents')
        latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        assert os.path.exists(latents_path)
    else:
        latents_path = None
        
    ## do noise rectification ##
    if args.input_image_path != None:
        assert os.path.exists(args.input_image_path)
        image_type = args.input_image_path.split('.')[-1]
        input_image = Image.open(args.input_image_path)
        print(input_image.format, input_image.size, input_image.mode)
        # args.height, args.width = 304, 512
        assert isinstance(input_image, Image.Image)
        
        if args.noise_rectification_period == None:
            args.noise_rectification_period = torch.tensor([0, 0.6])
    else:
        input_image = None
        

    # =========================================
    # ============= sample videos =============
    # =========================================

    video_frames = inference(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        xformers=args.xformers,
        sdp=args.sdp,
        spatial_lora_path=args.spatial_path_folder,
        temporal_lora_path=args.temporal_path_folder,
        lora_rank=args.lora_rank,
        spatial_lora_scale=args.spatial_path_scale,
        temporal_lora_scale=args.temporal_path_scale,
        seed=args.seed,
        latents_path=latents_path,
        noise_prior=args.noise_prior,
        repeat_num=args.repeat_num, 
        out_name=out_name,
        input_image=input_image,
        noise_rectification_period=args.noise_rectification_period,
        noise_rectification_weight=args.noise_rectification_weight,
        noise_rectification_weight_start_omega=args.noise_rectification_weight_start_omega,
        noise_rectification_weight_end_omega=args.noise_rectification_weight_end_omega,
    )


