import torch
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import DDIMScheduler, TextToVideoSDPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet3DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
# from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
# from diffusers.utils.constants import USE_PEFT_BACKEND
from PIL import Image
import numpy as np
from einops import rearrange
from diffusers.pipelines.text_to_video_synthesis import TextToVideoSDPipelineOutput
import os
import torchvision.transforms as T
from diffusers.utils import randn_tensor
from pathlib import Path
import yaml

from .injection_utils import register_conv_injection, register_time, register_spatial_attention_injection, register_temporal_attention_injection, load_source_latents_t, load_source_latents_T

def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images

def preprocess_image(image, width, height):
    assert isinstance(image, Image.Image)
    image = np.array(image.resize((width, height))).astype(np.float32) / 255.0
    image = np.expand_dims(image, 0)
    image = image.transpose(0, 3, 1, 2)
    image = 2.0 * image - 1.0
    image = torch.from_numpy(image)
    return image

def pil_to_numpy(images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

class InjectionPipeline(TextToVideoSDPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        
    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def get_ddim_eps(self, src_noise_path, latent):
        noise_step, noisy_latent = load_source_latents_T(src_noise_path)
        # print('size check in get_ddim_eps:', noisy_latent.shape, latent.shape)
        alpha_prod_T = self.scheduler.alphas_cumprod[noise_step]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        eps = (noisy_latent - mu_T * latent) / sigma_T
        return eps

    # get src video frames and encode them to latents
    @torch.no_grad()
    def get_data(self, src_frames_path, src_noise_path, mask_frames_path, num_frames, width, height, device, dtype, generator):
        
        # # load frames
        # paths = [os.path.join(src_frames_path, "%05d.jpg" % idx) for idx in range(num_frames)]
        # if not os.path.exists(paths[0]):
        #     paths = [os.path.join(src_frames_path, "%05d.png" % idx) for idx in range(num_frames)]
        # frames = [Image.open(paths[idx]).convert('RGB') for idx in range(num_frames)]
        # # if frames[0].size[0] == frames[0].size[1]:
        # #     frames = [frame.resize((width, height), resample=Image.Resampling.LANCZOS) for frame in frames]
        # # TODO : check float16
        # frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(dtype).to(device) #float16

        # # encode to latents
        # latents = self.vae.encode(frames).latent_dist.sample().mul_(0.18215)
        # latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1).contiguous()
        
        frames, latents = self.get_image_data(
                src_frames_path,
                num_frames,
                width,
                height,
                device,
                dtype,
                generator,
            )
        # print(f'----> get_data dtype check : {dtype}') #float32
        
        # get noise
        eps = self.get_ddim_eps(src_noise_path, latents).to(dtype).to(device) #float16
        
        # load mask (load one images first)
        if mask_frames_path == "":
            mask = None
        elif os.path.isdir(mask_frames_path):
            mask_paths = [os.path.join(mask_frames_path, "%05d.jpg" % idx) for idx in range(num_frames)]
            if not os.path.exists(mask_paths[0]):
                mask_paths = [os.path.join(mask_frames_path, "%05d.png" % idx) for idx in range(num_frames)]
                
            mask = [Image.open(mask_paths[idx]).convert("L") for idx in range(num_frames)]
            
            mask = pil_to_numpy(mask)  # to np
            mask = numpy_to_pt(mask)  # to pt
            
            # binarize
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
        else:
            mask = Image.open(mask_frames_path).convert("L") # to grayscale
            
            mask = pil_to_numpy(mask)  # to np
            mask = numpy_to_pt(mask)  # to pt
            
            # binarize
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
        
        return frames, latents, eps, mask
    
    # get a single guidance images and encode it to latent
    # TODO : use original video frames directly !
    @torch.no_grad()
    def get_image_data(self, guidance_img_path, num_frames, width, height, device, dtype, generator):
        if guidance_img_path == "":
            return None
        
        if os.path.isdir(guidance_img_path):
            # load frames
            paths = [os.path.join(guidance_img_path, "%05d.jpg" % idx) for idx in range(num_frames)]
            if not os.path.exists(paths[0]):
                paths = [os.path.join(guidance_img_path, "%05d.png" % idx) for idx in range(num_frames)]
            
            frame = [Image.open(paths[idx]) for idx in range(num_frames)]
            frame = torch.cat([preprocess_image(f, width, height) for f in frame], dim=0)
            frame = frame.to(device=device, dtype=dtype)
            
            print(f"get_image_data : frame.shape = {frame.shape}")
            
            # frame = torch.stack([T.ToTensor()(frame) for frame in frame]).to(torch.float32).to(device)
            # print(f"get_image_data : frame.shape = {frame.shape}")
            
            latent = self.vae.encode(frame).latent_dist.sample(generator).mul_(0.18215)
            
            latent = rearrange(latent, '(b f) c h w -> b c f h w', b = 1, f = num_frames) #[1,4,1,38,64]
            print('get_image_data, latent.shape =', latent.shape)
        else:
            # load frame
            frame = Image.open(guidance_img_path)
            frame = preprocess_image(frame, width, height)
            frame = frame.to(device=device, dtype=dtype)

            latent = self.vae.encode(frame).latent_dist.sample(generator).mul_(0.18215)
            
            latent = rearrange(latent, '(b f) c h w -> b c f h w', b = 1, f = 1) #[1,4,1,38,64]
            print('get_image_data, latent.shape =', latent.shape)
        
        #-----------------------------------------
        
        # frame = Image.open(guidance_img_path).convert('RGB').resize((width, height))
        # print('get_single_image_data, frame.shape(Image) =', frame)
        # # TODO : check float16
        # frame = torch.stack([T.ToTensor()(frame)]).to(torch.float16).to(device)
        # print('get_single_image_data, frame.shape =', frame.shape)

        # # encode to latents
        # latent = self.vae.encode(frame).latent_dist.sample().mul_(0.18215)
        # latent = rearrange(latent, "(b f) c h w -> b c f h w", b=1).contiguous()
        # print('get_single_image_data, latent.shape =', latent.shape)
        
        #-----------------------------------------
        
        return frame, latent

    def init_injection(self, config):
        self.scheduler.set_timesteps(config.num_steps, device=config.device)
        
        # conv_injection_t = int(config.n_steps * config.pnp_f_t)
        attnS_injection_t = int(config.num_steps * config.pnp_attnS_t)
        attnC_injection_t = int(config.num_steps * config.pnp_attnC_t)
        attntmp_injection_t = int(config.num_steps * config.pnp_attntmp_t)
        print(attnS_injection_t, attnC_injection_t, attntmp_injection_t)
        
        # self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        self.attnS_injection_timesteps = self.scheduler.timesteps[:attnS_injection_t] if attnS_injection_t >= 0 else []
        self.attnC_injection_timesteps = self.scheduler.timesteps[attnC_injection_t:] if attnC_injection_t < len(self.scheduler.timesteps) and attnC_injection_t >= 0 else []
        self.attntmp_injection_timesteps = self.scheduler.timesteps[attntmp_injection_t:] if attntmp_injection_t < len(self.scheduler.timesteps) and attntmp_injection_t >= 0 else []
        
        if len(self.attnS_injection_timesteps) != 0:
            print(f'[State] use spatial attention injection : t = [0-{len(self.attnS_injection_timesteps)-1}]')
        else:
            print(f'[State] no use spatial attention injection')
            
        if len(self.attnC_injection_timesteps) != 0:
            print(f'[State] use content attention injection : t = [{attnC_injection_t}-{len(self.scheduler.timesteps)-1}]')
        else:
            print(f'[State] no use spatial attention injection')
            
        if len(self.attntmp_injection_timesteps) != 0:
            print(f'[State] use temporal attention injection : t = [{attntmp_injection_t}-{len(self.scheduler.timesteps)-1}]')
        else:
            print(f'[State] no use temporal attention injection')
        
        # register_conv_injection(self, self.conv_injection_timesteps) # not much of impact
        register_spatial_attention_injection(self, self.attnS_injection_timesteps, self.attnC_injection_timesteps)
        register_temporal_attention_injection(self, self.attntmp_injection_timesteps)

    # add eps as parameter
    @torch.no_grad()
    def prepare_latents(self, eps, timesteps, batch_size, num_channels_latents, video_length,
                        height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor) # (b,c,f,h,w)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
            
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            # latents = self.scheduler.add_noise(latents, eps, timesteps[0]) ### add rand noise ??
        else:
            # TODO : excpetion when the target size is different with inversion size
            assert (latents.shape[3] == shape[3]) or (latents.shape[4] == shape[4])
            latents = latents.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            # noisy_latents
            latents = self.scheduler.add_noise(latents, eps, timesteps[0])

        return latents

    # without deal with masked_image_latents
    @torch.no_grad()
    def prepare_mask(
        self, mask, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        # deal with the case that there are many masks
        if mask.shape[0] != 1:
            mask = rearrange(mask, '(b f) c h w -> b c f h w', b=1)

        # TODO : dim = 9 need this
        # mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        
        return mask

    @torch.no_grad()
    def prepare_mask_image_latents(
        self, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return masked_image_latents
    
    @torch.no_grad()
    def get_ddim_inversion_prompt(self, exp_name, video_name):
        # TODO : get inversion prompt from ddim_latents_path
        inv_prompts_path = os.path.join('ddim-inversion', exp_name, "inversion_prompts.yaml")
        with open(inv_prompts_path, "r") as f:
            inv_prompts = yaml.safe_load(f)
        return inv_prompts[f"{video_name}"]
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def __call__(  
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # clip_skip: Optional[int] = None, # no use in diffusers 0.18.0
        exp_name: Optional[str] = "",
        src_frames_path: Optional[str] = "",
        src_noise_path: Optional[str] = "",
        video_name: Optional[str] = "",
        enable_injection: bool = False,
        mask_frames_path: Optional[str] = "",
        guidance_img_path: Optional[str] = "",
        enable_null_prompt: bool = False,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        print("__call__ device =", device)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        ### [diffusers 0.18.0] : if do_classifier_free_guidance
        #    ,return torch.cat([negative_prompt_embeds, prompt_embeds]) 
        # prompt_embeds = self._encode_prompt( 
        #     prompt,
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     lora_scale=text_encoder_lora_scale,
        #     # clip_skip=clip_skip,
        # )
        
        # prompt_embeds, negative_prompt_embeds = self.get_text_embeds(prompt, negative_prompt).chunk(2, dim=0)
        prompt_embeds, negative_prompt_embeds = self.get_text_embeds(prompt, negative_prompt).chunk(2, dim=0)
        prompt_embeds, negative_prompt_embeds = prompt_embeds.to(device), negative_prompt_embeds.to(device)
        print('prompt :',prompt[0], prompt_embeds.shape)
        print('negative_prompt :', negative_prompt[0], negative_prompt_embeds.shape)
        
        if enable_injection:
            ddim_inversion_prompt = self.get_ddim_inversion_prompt(exp_name, video_name)

            self.ddim_inversion_embeds, null_prompt_embeds = self.get_text_embeds(ddim_inversion_prompt, "").chunk(2, dim=0)
            self.ddim_inversion_embeds, null_prompt_embeds = self.ddim_inversion_embeds.to(device), null_prompt_embeds.to(device)
            print('ddim_inversion_prompt :', ddim_inversion_prompt, self.ddim_inversion_embeds.shape)
                

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # print('timesteps:',timesteps)

        # 5. Prepare latent variables
        self.frames, self.latents, self.eps, self.mask = self.get_data(
            src_frames_path,
            src_noise_path,
            mask_frames_path, # new for mask
            num_frames,
            width,
            height,
            device,
            prompt_embeds.dtype,
            generator,
        )
        print('-------------')
        print('self.latents.shape =', self.latents.shape)
        print('self.eps.shape =', self.eps.shape)
        print('self.frames.shape =', self.frames.shape)
        print('--------------')
        if self.mask != None:
            print('get_data mask size check :', self.mask.shape)
        else:
            print('get_data mask is None!!!!!')

        # get guidance image latent if provided
        if guidance_img_path != "":
            self.guidance_img_latent = self.get_image_data(
                guidance_img_path,
                num_frames,
                width,
                height,
                device,
                prompt_embeds.dtype,
                generator,
            )

        num_channels_latents = self.unet.config.in_channels
        
        # noisy_latents = original_latent + (eps from ddimVersion_T)
        if enable_injection: 
            noisy_latents = self.prepare_latents(
                self.eps,
                timesteps,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                self.latents, ########## None
            )
        else: # noisy_latents = original_latent + rand_noise
            noisy_latents = self.prepare_latents(
                self.eps,
                timesteps,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                None, # latents = None
            )
        self.latents = self.latents.to(device)
        print("latents to device:", self.latents.get_device())
        
        # get mask if provided
        if self.mask != None:
            self.mask = self.prepare_mask(
                self.mask,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )
            print('check mask after prepare_mask, mask.shape =', self.mask.shape) # mask size error
        
        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        # TODO : denoising for every batch_size frames
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if enable_injection:
                    # register the time step and features in pnp injection modules
                    source_latents = load_source_latents_t(t, src_noise_path).to(device)
                    
                    if enable_null_prompt: # input 4 things
                        latent_model_input = torch.cat([source_latents] + ([noisy_latents] * 3)) # no prompt in last channel
                        text_embed_input = torch.cat([self.ddim_inversion_embeds, prompt_embeds, negative_prompt_embeds, prompt_embeds], dim=0)
                    else:
                        latent_model_input = torch.cat([source_latents] + ([noisy_latents] * 2))
                        text_embed_input = torch.cat([self.ddim_inversion_embeds, prompt_embeds, negative_prompt_embeds], dim=0)
                    
                    register_time(self, t.item())
                else:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    text_embed_input = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=text_embed_input,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if enable_injection:
                    if enable_null_prompt:
                        _, noise_pred_uncond, noise_pred_text, _ = noise_pred.chunk(4)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                elif do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = noisy_latents.shape
                noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                noisy_latents = noisy_latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)


                # mask
                if self.mask != None and self.guidance_img_latent != None:
                    # init_latents_next_t = self.guidance_img_latent.repeat(1, 1, num_frames, 1, 1)
                    init_latents_next_t = self.guidance_img_latent
                    
                    if i < len(timesteps) - 1:
                        init_latents_next_t = self.scheduler.add_noise(
                            self.latents, self.eps, torch.tensor([timesteps[i+1]])
                        )
                    # noisy_latents = (1 - self.mask) * init_latents_next_t + self.mask * noisy_latents # keep foreground
                    noisy_latents = ((self.mask) * init_latents_next_t) + (1 - self.mask) * noisy_latents # keep background


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, noisy_latents)

        # 8. Post processing
        if output_type == "latent":
            return TextToVideoSDPipelineOutput(frames=noisy_latents)

        video_tensor = self.decode_latents(noisy_latents) #### image [1024,512] out of memory

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        # 9. Offload all models
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (video,)

        return TextToVideoSDPipelineOutput(frames=video)