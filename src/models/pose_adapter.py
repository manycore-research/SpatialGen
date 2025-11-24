# Copyright (c) 2023-2024, Chuan Fang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from accelerate.logging import get_logger
from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config, FrozenDict
from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention, XFormersAttnProcessor
from src.utils.typing import *

# helpers
logger = get_logger(__name__)

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


# class CustomTransformer(nn.Module):
class CustomTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Attention(dim, heads = heads, dim_head = dim_head),
                Attention(
                        query_dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=0.0,
                        bias=False,
                        cross_attention_dim=None,
                        upcast_attention=False,
                        processor=XFormersAttnProcessor(),
                    ),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class RayMapEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, 
                 image_size: int = 512,
                 patch_size: int = 8,
                 in_channel: int=6,
                 out_channel: int = 16,
                 inter_dims: int = 384,
                 transformer_layers: int = 1,
                 transformer_heads: int = 6,
                 transformer_dim_head: int = 64,
                 transformer_mlp_dim: int = 384,
                 **kwargs):
        
        # super(RayMapEncoder, self).__init__(config)
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # print(f"[RayMapEncoder] image_size: {image_height} x {image_width}, patch_size: {patch_height} x {patch_width}")
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = in_channel * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, inter_dims),
            nn.LayerNorm(inter_dims),
        )

        self.patched_h, self.patched_w = image_height // patch_height, image_width // patch_width
        self.patched_embed_dim = inter_dims
         

        self.transformer = CustomTransformer(dim=inter_dims, 
                                       depth=transformer_layers, 
                                       heads=transformer_heads, 
                                       dim_head=transformer_dim_head, 
                                       mlp_dim=transformer_mlp_dim)

        self.linear_head = nn.Linear(inter_dims, out_channel)

    @property
    def pos_embedding(self):
        return posemb_sincos_2d(
            h = self.patched_h,
            w = self.patched_w,
            dim = self.patched_embed_dim,
        )
    @classmethod
    def from_pretrained_new(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        
        config_path = pretrained_model_name_or_path
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        from diffusers import __version__
        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }
        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )
        init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)

        # Initialize model with the config
        model = cls(**init_dict)
        
        # Determine model file path
        model_dir = pretrained_model_name_or_path + "/ray_encoder"
        print(f"Loading model from {model_dir}")
        if os.path.isdir(model_dir):
            if os.path.exists(os.path.join(model_dir, "diffusion_pytorch_model.safetensors")):
                safetensors_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
            else:
                safetensors_path = os.path.join(model_dir, "model.safetensors")
            bin_path = os.path.join(model_dir, "pytorch_model.bin")
            
            # Try to load state dict (prefer safetensors if available)
            state_dict = None
            if os.path.exists(safetensors_path):
                try:
                    from safetensors.torch import load_file  # Optional dependency
                    state_dict = load_file(safetensors_path)
                except ImportError:
                    raise ImportError(
                        "`safetensors` is required to load .safetensors files. "
                        "Install with `pip install safetensors`."
                    )
            elif os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {model_dir}. "
                    "Expected either 'model.safetensors' or 'pytorch_model.bin'."
                )
        else:
            raise NotADirectoryError(f"{model_dir} is not a directory")

        # Handle position embedding mismatch
        current_state_dict = model.state_dict()
        if "pos_embedding" in state_dict:
            pretrained_pos_embed = state_dict["pos_embedding"]
            current_pos_embed = current_state_dict["pos_embedding"]
            
            if pretrained_pos_embed.shape != current_pos_embed.shape:
                print(
                    f"Ignoring pretrained pos_embedding due to shape mismatch "
                    f"({pretrained_pos_embed.shape} vs {current_pos_embed.shape}). "
                    "Using newly initialized position embeddings."
                )
                del state_dict["pos_embedding"]

        # Load state dict with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Log warnings
        if missing_keys:
            print(f"Missing keys in pretrained model: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in pretrained model: {unexpected_keys}")

        return model

    @classmethod
    def from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs):
        r"""
        Instantiate a Python class from a config dictionary.

        Parameters:
            config (`Dict[str, Any]`):
                A config dictionary from which the Python class is instantiated. Make sure to only load configuration
                files of compatible classes.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it is loaded) and initiate the Python class.
                `**kwargs` are passed directly to the underlying scheduler/model's `__init__` method and eventually
                overwrite the same named arguments in `config`.

        Returns:
            [`ModelMixin`] or [`SchedulerMixin`]:
                A model or scheduler object instantiated from a config dictionary.

        Examples:

        ```python
        >>> from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler

        >>> # Download scheduler from huggingface.co and cache.
        >>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

        >>> # Instantiate DDIM scheduler class with same config as DDPM
        >>> scheduler = DDIMScheduler.from_config(scheduler.config)

        >>> # Instantiate PNDM scheduler class with same config as DDPM
        >>> scheduler = PNDMScheduler.from_config(scheduler.config)
        ```
        """
        # <===== TO BE REMOVED WITH DEPRECATION
        # TODO(Patrick) - make sure to remove the following lines when config=="model_path" is deprecated
        if "pretrained_model_name_or_path" in kwargs:
            config = kwargs.pop("pretrained_model_name_or_path")

        if config is None:
            raise ValueError("Please make sure to provide a config as the first positional argument.")
        # ======>

        if not isinstance(config, dict):
            deprecation_message = "It is deprecated to pass a pretrained model name or path to `from_config`."
            if "Scheduler" in cls.__name__:
                deprecation_message += (
                    f"If you were trying to load a scheduler, please use {cls}.from_pretrained(...) instead."
                    " Otherwise, please make sure to pass a configuration dictionary instead. This functionality will"
                    " be removed in v1.0.0."
                )
            elif "Model" in cls.__name__:
                deprecation_message += (
                    f"If you were trying to load a model, please use {cls}.load_config(...) followed by"
                    f" {cls}.from_config(...) instead. Otherwise, please make sure to pass a configuration dictionary"
                    " instead. This functionality will be removed in v1.0.0."
                )
            deprecate("config-passed-as-path", "1.0.0", deprecation_message, standard_warn=False)
            config, kwargs = cls.load_config(pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)

        # Allow dtype to be specified on initialization
        if "dtype" in unused_kwargs:
            init_dict["dtype"] = unused_kwargs.pop("dtype")

        # add possible deprecated kwargs
        for deprecated_kwarg in cls._deprecated_kwargs:
            if deprecated_kwarg in unused_kwargs:
                init_dict[deprecated_kwarg] = unused_kwargs.pop(deprecated_kwarg)

        # Return model and optionally state and/or unused_kwargs
        model = cls(**init_dict)

        # make sure to also save config parameters that might be used for compatible classes
        # update _class_name
        if "_class_name" in hidden_dict:
            hidden_dict["_class_name"] = cls.__name__

        model.register_to_config(**hidden_dict)

        # add hidden kwargs of compatible classes to unused_kwargs
        unused_kwargs = {**unused_kwargs, **hidden_dict}

        if return_unused_kwargs:
            return (model, unused_kwargs)
        else:
            return model
    
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
    
    def forward(self, x: Float[Tensor, "B T C H W"], **kwargs) -> Float[Tensor, "B T C H W"]:
        device = x.device
        bs, fs = x.shape[0], x.shape[1]
        x = rearrange(x, "B T C H W -> (B T) C H W")
        # logger.info(f'input: {x.shape}')
        x = self.to_patch_embedding(x)
        # logger.info(f'self.pos_embedding: {self.pos_embedding.shape}')
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        # logger.info(f'transformer output: {x.shape}')
        
        output = self.linear_head(x)
        # logger.info(f'output : {output.shape}')
        output = rearrange(output, 'b (h w) c -> b c h w', h=self.patched_h, w=self.patched_w)
        output = rearrange(output, "(B T) C H W -> B T C H W", B=bs, T=fs)
        
        return output

