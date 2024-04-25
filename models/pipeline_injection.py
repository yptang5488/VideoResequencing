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

from .injection_utils import register_conv_injection

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
    
    def init_injection(self, config):
        attn_injection_t = int(config.n_steps * config.pnp_f_t)
        conv_injection_t = int(config.n_steps * config.pnp_f_t)
        self.attn_injection_timesteps = self.scheduler.timesteps[:attn_injection_t] if attn_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        # register_extended_attention_pnp(self, self.attn_injection_timesteps)
        register_conv_injection(self, self.conv_injection_timesteps)

    '''
    # add input_image as parameter
    def prepare_latents(self, input_image, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
            
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # Our method first adds noise to the input image and keep the added noise for latter rectification.
        noise = latents.clone()
        input_image = preprocess_image(input_image, width, height)
        input_image = input_image.to(device=device, dtype=dtype)

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(input_image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(input_image).latent_dist.sample(generator)
        
        init_latents = rearrange(init_latents, '(b f) c h w -> b c f h w', b = batch_size, f = 1) #[1,4,1,38,64]
        
        ## defualt
        init_latents = init_latents.repeat((1, 1, video_length, 1, 1)) * 0.18215 #[1,4,16,38,64]
        noisy_latents = self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0])
        
        ## YP experiment
        # init_latents = init_latents.repeat((1, 1, video_length//2, 1, 1)) * 0.18215 #[1,4,16,38,64]
        # init_latents_combine = torch.cat((init_latents, noise[:, :, video_length//2:video_length, :, :]), dim=2)
        # print("combine shape :", init_latents_combine.shape)
        # noisy_latents = self.scheduler.add_noise(init_latents_combine, noise, self.scheduler.timesteps[0])

        return noisy_latents, noise

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
        
        input_image = None,
        noise_rectification_period: Optional[list] = None,
        noise_rectification_weight: Optional[torch.Tensor] = None,
        noise_rectification_weight_start_omega: float = 1.0,
        noise_rectification_weight_end_omega: float = 0.5,

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
        prompt_embeds = self._encode_prompt( 
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            # clip_skip=clip_skip,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        noisy_latents, noise = self.prepare_latents(
            input_image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents = latents.to(device)
        print("latents to device:", latents.get_device())
        
        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        noise_rectification_weight = torch.cat([torch.linspace(noise_rectification_weight_start_omega, noise_rectification_weight_end_omega, num_frames//2), 
                                                torch.linspace(noise_rectification_weight_end_omega, noise_rectification_weight_end_omega, num_frames//2)])
        print('Before denoiseing loop, noise_rectification_weight =', noise_rectification_weight)
        

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # [The core code of noise rectification method.]
                # our method rectifies the predicted noise with the GT noise to realize image-to-video.
                if noise_rectification_period is not None:
                    assert len(noise_rectification_period) == 2
                    if noise_rectification_weight is None:
                        # default
                        noise_rectification_weight = torch.cat([torch.linspace(noise_rectification_weight_start_omega, noise_rectification_weight_end_omega, num_frames//2), 
                                                                torch.linspace(noise_rectification_weight_end_omega, noise_rectification_weight_end_omega, num_frames//2)])
                        
                        # inverse proportion
                        # noise_rectification_weight = torch.linspace(noise_rectification_weight_start_omega, noise_rectification_weight_end_omega, num_frames)
                        
                    noise_rectification_weight = noise_rectification_weight.view(1, 1, num_frames, 1, 1)
                    noise_rectification_weight = noise_rectification_weight.to(latent_model_input.dtype).to(latent_model_input.device)

                    if i >= len(timesteps) * noise_rectification_period[0] and i < len(timesteps) * noise_rectification_period[1]:
                        delta_frames = noise - noise_pred
                        delta_noise_adjust = noise_rectification_weight * (delta_frames[:,:,[0],:,:].repeat((1, 1, num_frames, 1, 1))) + \
                                            (1 - noise_rectification_weight) * delta_frames
                        noise_pred = noise_pred + delta_noise_adjust # YP modify here

                # reshape latents
                bsz, channel, frames, width, height = noisy_latents.shape
                noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                
                # compute the previous noisy sample x_t -> x_t-1
                noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                noisy_latents = noisy_latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

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
    '''