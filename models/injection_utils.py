import argparse
import os
import platform
import re
import warnings
from typing import Optional, Union, List
import torch.nn.functional as F

import torch
from diffusers import DDIMScheduler, TextToVideoSDPipeline
from diffusers.models.attention_processor import Attention
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
import random
import glob

from MotionDirector_train import export_to_video, handle_memory_attention, load_primary_models, unet_and_text_g_c, freeze_models
from utils.lora_handler import LoraHandler
from utils.ddim_utils import ddim_inversion
import imageio

from PIL import Image
from models.pipeline_I2V_NoiseRect_zeroscope import NoiseRectSDPipeline
import inspect

def load_source_latents_t(t, src_noise_path):
    path_t = os.path.join(src_noise_path, f"noisy_latents_{t}.pt")
    assert os.path.exists(path_t), f"Missing latents at t {t} path {path_t}"
    ddim_latents_at_t = torch.load(path_t)
    return ddim_latents_at_t

def load_source_latents_T(src_noise_path):
    noise_step_T = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(src_noise_path, f'noisy_latents_*.pt'))])
    path_t = os.path.join(src_noise_path, f'noisy_latents_{noise_step_T}.pt')
    noisy_latent_T = torch.load(path_t)
    return noise_step_T, noisy_latent_T


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    
    # not use pnp !
    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         setattr(module.attn1, 't', t)
    
    # down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            # module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            # setattr(module, 't', t)
            module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1
            setattr(module, "t", t)
            
    # for res in down_res_dict:
    #     for block in down_res_dict[res]:
    #         module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
    #         setattr(module, 't', t)
    #         module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
    #         setattr(module, 't', t)
            
    # module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    # setattr(module, 't', t)
    # module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    # setattr(module, 't', t)

def register_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb): # ResnetBlock2D
            hidden_states = input_tensor

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm1(hidden_states, temb)
            else:
                hidden_states = self.norm1(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = self.time_emb_proj(temb)[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm2(hidden_states, temb)
            else:
                hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            
            ## implement injection ##
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                # print('implement injection in conv')
                n_frames = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[n_frames:2 * n_frames] = hidden_states[:n_frames]
                # inject conditional
                hidden_states[2 * n_frames:] = hidden_states[:n_frames]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1] # CrossAttnUpBlock3D / ResnetBlock2D
    # print('-----------------------------------------------------------')
    # print('register_conv_injection() : conv_module =', conv_module)
    # print('-----------------------------------------------------------')
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
    
def register_spatial_attention_injection(model, spatial_injection_schedule, content_injection_schedule, d_s=0.1, d_t=0.5):
    def sa_s_forward(self): # BasicTransformerBlock.attn1 (Attention), have checked that all are built by AttnProcessor2_0
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            inner_dim = hidden_states.shape[-1]

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            ## implement injection ##
            # if hasattr(self, "spatial_injection_schedule") and hasattr(self, t):
            if self.spatial_injection_schedule is not None and self.t is not None:
                if self.t in self.spatial_injection_schedule or self.t == 1000: # TODO : why 1000?
                    # print(f"at timestemp {self.t}: implement spatial_injection_schedule")
                    n_frames = int(hidden_states.shape[0] // 4)
                    
                    # inject source into unconditional
                    query[n_frames : 2 * n_frames] = query[:n_frames]
                    key[n_frames : 2 * n_frames] = key[:n_frames]

                    # inject source into conditional
                    query[2 * n_frames : 3 * n_frames] = query[:n_frames]
                    key[2 * n_frames : 3 * n_frames] = key[:n_frames]
                    
                    # # inject source into motion branch
                    # query[3 * n_frames :] = query[:n_frames]
                    # key[3 * n_frames :] = key[:n_frames]
            
            if self.content_injection_schedule is not None and self.t is not None:    
                if self.t in self.content_injection_schedule:
                    # print(f"at timestemp {self.t}: implement content_injection_schedule")
                    n_frames = int(hidden_states.shape[0] // 4)
                    
                    # inject source into unconditional
                    value[n_frames : 2 * n_frames] = value[:n_frames]

                    # inject source into conditional
                    value[2 * n_frames : 3 * n_frames] = value[:n_frames]


            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward
    
    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         # print('BasicTransformerBlock forward modify')
    #         module.attn1.forward = sa_s_forward(module.attn1)
    #         setattr(module.attn1, "injection_schedule", [])  # Disable PNP
    
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}

    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            # print('-----------------------------------------------------------')
            # print(f'[change by res_dict] model.unet.up_blocks =', model.unet.up_blocks.__class__.__name__)
            # print(f'[change by res_dict] model.unet.up_blocks[{res}].attentions[{block}] =', model.unet.up_blocks[res].attentions[block].__class__.__name__)
            # print(f'[change by res_dict] model.unet.up_blocks[{res}].attentions[{block}].transformer_blocks[0] =', model.unet.up_blocks[res].attentions[block].transformer_blocks[0].__class__.__name__)
            # print(f'[change by res_dict] model.unet.up_blocks[{res}].attentions[{block}].transformer_blocks[0].attn1 =', model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.__class__.__name__)
            # print('-----------------------------------------------------------')
            module.forward = sa_s_forward(module)
            print(f'[res dict] upblocks[{res}].attentions[{block}].attn1 : {module.__class__.__name__}')
            setattr(module, "spatial_injection_schedule", spatial_injection_schedule)
            setattr(module, "content_injection_schedule", content_injection_schedule)


def register_temporal_attention_injection(model, injection_schedule, d_s=0.1, d_t=0.5):
    def sa_t_forward(self): # BasicTransformerBlock.attn (Attention), have checked that all are built by AttnProcessor2_0
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            inner_dim = hidden_states.shape[-1]

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            ## implement injection ##
            if self.injection_schedule is not None and self.t is not None:
                if self.t in self.injection_schedule:
                    # print(f"at timestemp {self.t}: implement temporal_injection_schedule")
                    n_frames = int(hidden_states.shape[0] // 4)
                    
                    # inject source into unconditional
                    query[n_frames : 2 * n_frames] = query[3 * n_frames:]
                    key[n_frames : 2 * n_frames] = key[3 * n_frames:]

                    # inject source into conditional
                    query[2 * n_frames : 3 * n_frames] = query[3 * n_frames:]
                    key[2 * n_frames : 3 * n_frames] = key[3 * n_frames:]

            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward
    
    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         print('BasicTransformerBlock temp_attentions forward modify')
    #         # module.attn2.forward = sa_t_forward(module.attn2)
    #         setattr(module.attn2, "injection_schedule", [])  # Disable PNP
    
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}

    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1
            print(f'[res dict] upblocks[{res}].attentions[{block}].attn1 : {module.__class__.__name__}')
            # print('-----------------------------------------------------------')
            # print(f'[change by res_dict] model.unet.up_blocks =', model.unet.up_blocks.__class__.__name__)
            # print(f'[change by res_dict] model.unet.up_blocks[{res}].attentions[{block}] =', model.unet.up_blocks[res].temp_attentions[block].__class__.__name__)
            # print(f'[change by res_dict] model.unet.up_blocks[{res}].attentions[{block}].transformer_blocks[0] =', model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].__class__.__name__)
            # print(f'[change by res_dict] model.unet.up_blocks[{res}].attentions[{block}].transformer_blocks[0].attn1 =', model.unet.up_blocks[res].temp_attentions[block].transformer_blocks[0].attn1.__class__.__name__)
            # print('-----------------------------------------------------------')
            module.forward = sa_t_forward(module)
            setattr(module, "injection_schedule", injection_schedule)

