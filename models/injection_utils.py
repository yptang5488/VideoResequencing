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
            
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1] # CrossAttnUpBlock3D -> ResnetBlock2D
    print('-----------------------------------------------------------')
    print('register_conv_injection() : conv_module =', conv_module)
    print('-----------------------------------------------------------')
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)