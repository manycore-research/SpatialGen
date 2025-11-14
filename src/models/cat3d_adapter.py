import os
import os.path as osp

import torch
import torchvision
import numpy as np
from einops import rearrange
from icecream import ic
import open3d as o3d
from PIL import Image
from accelerate.logging import get_logger
import torch.nn.functional as tF

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config, FrozenDict
from diffusers.utils import deprecate

from src.utils.typing import *
from src.models.pose_adapter import RayMapEncoder, RayMapEncoderConfig
from diffusers_spatialgen import UNetMVMM2DConditionModel

logger = get_logger(__name__)




class CAT3DAdaptor(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, 
                 unet_config: Dict,
                 ray_encoder_config: Dict,
                 unet_type: str = "UNetMVMM2DConditionModel",
                 num_sample_views: int = 8,
                 prediction_types: List[str] = ["rgb", "depth"],
                 unet_in_channels: int = 21,
                 use_layout_prior: bool = False,
                 unet: Union[UNetMVMM2DConditionModel] = None, 
                 ray_encoder: RayMapEncoder = None):
        super().__init__()
                
        self.unet = unet
        self.ray_encoder = ray_encoder
        self.unet_type = unet_type
        self.unet_config = dict(unet_config)
        self.ray_encoder_config = dict(ray_encoder_config)
        self.num_sample_views = num_sample_views
        self.prediction_types = prediction_types
        self.use_layout_prior = use_layout_prior
        self.num_tasks = len(self.prediction_types) + 2 if self.use_layout_prior else len(self.prediction_types)
        self.unet_in_channels = unet_in_channels

        if self.unet_in_channels >= 21:
            self.ray_channel = 16
        elif self.unet_in_channels == 11:
            self.ray_channel = 6
            
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
        # model = cls(**init_dict)
        # Initialize the UNet ( UNetMVMM2DConditionModel )
        if init_dict["unet_type"] == "UNetMVMM2DConditionModel":
            unet = UNetMVMM2DConditionModel.from_config(init_dict["unet_config"])
        else:
            unet = PixArtTransformerMV2DModel.from_config(init_dict["unet_config"])
        
        # Initialize the ray encoder
        ray_encoder = RayMapEncoder.from_config(init_dict["ray_encoder_config"])

        # Create the CAT3DAdaptor instance
        model = cls(init_dict["unet_config"],
                   init_dict["ray_encoder_config"],
                   init_dict["unet_type"],
                   init_dict["num_sample_views"],
                   init_dict["prediction_types"],
                   init_dict["unet_in_channels"],
                   init_dict["use_layout_prior"],
                   unet, 
                   ray_encoder, 
                   **kwargs)

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
    
    @classmethod
    def _from_config(cls, config, **kwargs):
        # Initialize the UNet (either UNetMVMM2DConditionModel or PixArtTransformerMV2DModel)
        if config.unet_type == "UNetMVMM2DConditionModel":
            unet = UNetMVMM2DConditionModel.from_config(config.unet_config)
        else:
            unet = PixArtTransformerMV2DModel.from_config(config.unet_config)
        
        # Initialize the ray encoder
        ray_encoder = RayMapEncoder.from_config(config.ray_encoder_config)

        # Create the CAT3DAdaptor instance
        return cls(config.unet_config,
                   config.ray_encoder_config,
                   config.unet_type,
                   config.num_sample_views,
                   config.prediction_types,
                   config.unet_in_channels,
                   config.use_layout_prior,
                   unet, 
                   ray_encoder, 
                   **kwargs)
    
    def save_pretrained(self, save_directory, is_main_process = True, save_function = None, safe_serialization = True, variant = None, max_shard_size = "10GB", push_to_hub = False, **kwargs):
        # update config, filter out ray_encoder and unet object
        self.register_to_config(**{"ray_encoder": None, "unet": None})
        return super().save_pretrained(save_directory, is_main_process, save_function, safe_serialization, variant, max_shard_size, push_to_hub, **kwargs)
        
    def encode_cam_rays(self, cam_rays: Float[Tensor, "B Nt 6 H W"], drop_masks: Float[Tensor, "B "]) -> Float[Tensor, "BNt C H W"]:
        """
        Encodes camera rays using ray encoder
        params:
            cam_rays: camera rays, shape: BNt 6 H W
        returns:
            ray_latents: ray latents, shape: BNt C H W
        """
        input_rays = cam_rays.repeat(1, self.num_tasks, 1, 1, 1)  # (B, num_tasks*Nt, 6, H, W)
        if self.ray_encoder is not None:
            ray_latents = self.ray_encoder(input_rays)  # (B, num_tasks*Nt, 16, H//8, W//8)
        else:
            bsz = input_rays.shape[0]
            latent_size = self.unet.config.sample_size
            input_rays = rearrange(input_rays, "B Nt C H W -> (B Nt) C H W")  # (B, num_tasks*Nt, 6*H*W)
            ray_latents = tF.interpolate(input_rays, size=[latent_size, latent_size], mode="bilinear", align_corners=False)
            ray_latents = rearrange(ray_latents, "(B Nt) C H W -> B Nt C H W", B=bsz)  # (B, num_tasks*Nt, 6, H//8, W//8)
        # cfg drop ray embeds
        ray_latents = ray_latents * drop_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ray_latents = rearrange(ray_latents, "B Nt C H W -> (B Nt) C H W")  # (B*num_tasks*Nt, 16, H//8, W//8)
        return ray_latents

    def compose_rgbdns_latents(self, 
                               batch_data: Dict[str, Any], 
                               weight_dtype: torch.dtype, 
                               vae: AutoencoderKL) -> Float[Tensor, "Bt C H W"]:
        
        target_view_imgs = torch.cat([
                    batch_data["image_target"],
                    batch_data["depth_target"],
                    batch_data["normal_target"],
                    batch_data["semantic_target"]], dim=1).to(dtype=weight_dtype)  # 4B,T,3,H,W
        target_view_imgs = rearrange(target_view_imgs, "b t c h w -> (b t) c h w")  # B*num_tasks*(T_out), 3, H, W
        logger.info(f"target_view_imgs shape: {target_view_imgs.shape}")
        target_view_latents: Float[Tensor, "BNt C H W"] = vae.encode(target_view_imgs).latent_dist.sample() * vae.config.scaling_factor
        
        input_depth = batch_data["depth_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_depth = rearrange(input_depth, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_depth_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_depth).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8

        input_normal = batch_data["normal_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_normal = rearrange(input_normal, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_normal_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_normal).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
        
        input_semantic = batch_data["semantic_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_semantic = rearrange(input_semantic, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_semantic_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_semantic).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
        prediction_latents = torch.cat([target_view_latents, input_depth_latents, input_normal_latents, input_semantic_latents], dim=0)
        
        return prediction_latents
    
    def compose_rgbdn_latents(self, 
                              batch_data: Dict[str, Any], 
                              weight_dtype: torch.dtype, 
                              vae: AutoencoderKL) -> Float[Tensor, "Bt C H W"]:
        
        target_view_imgs = torch.cat([
                batch_data["image_target"],
                batch_data["depth_target"],
                batch_data["normal_target"]], dim=1).to(dtype=weight_dtype)  # 3B,T,3,H,W
        target_view_imgs = rearrange(target_view_imgs, "b t c h w -> (b t) c h w")  # B*num_tasks*(T_out), 3, H, W
        logger.info(f"target_view_imgs shape: {target_view_imgs.shape}")
        target_view_latents: Float[Tensor, "BNt C H W"] = vae.encode(target_view_imgs).latent_dist.sample() * vae.config.scaling_factor

        input_depth = batch_data["depth_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_depth = rearrange(input_depth, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_depth_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_depth).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8

        input_normal = batch_data["normal_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_normal = rearrange(input_normal, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_normal_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_normal).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
        prediction_latents = torch.cat([target_view_latents, input_depth_latents, input_normal_latents], dim=0)
        
        return prediction_latents
    
    def compose_rgbds_latents(self, 
                              batch_data: Dict[str, Any], 
                              weight_dtype: torch.dtype, 
                              vae: AutoencoderKL) -> Float[Tensor, "Bt C H W"]:
        
        target_view_imgs = torch.cat([
                batch_data["image_target"],
                batch_data["depth_target"],
                batch_data["semantic_target"]], dim=1).to(dtype=weight_dtype)  # 3B,T,3,H,W
        target_view_imgs = rearrange(target_view_imgs, "b t c h w -> (b t) c h w")  # B*num_tasks*(T_out), 3, H, W
        target_view_latents: Float[Tensor, "BNt C H W"] = vae.encode(target_view_imgs).latent_dist.sample() * vae.config.scaling_factor

        input_depth = batch_data["depth_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_depth = rearrange(input_depth, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_depth_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_depth).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8

        input_semantic = batch_data["semantic_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_semantic = rearrange(input_semantic, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_sem_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_semantic).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
        prediction_latents = torch.cat([target_view_latents, input_depth_latents, input_sem_latents], dim=0)
        
        return prediction_latents
        
    def compose_rgbd_latents(self,
                                batch_data: Dict[str, Any],
                                weight_dtype: torch.dtype,
                                vae: AutoencoderKL,
                                depth_vae: AutoencoderKL = None) -> Float[Tensor, "Bt C H W"]:
        if depth_vae is None:
            target_view_imgs = torch.cat([batch_data["image_target"], 
                                        batch_data["depth_target"]], dim=1).to(dtype=weight_dtype)  # 2B,T,3,H,W
            target_view_imgs = rearrange(target_view_imgs, "b t c h w -> (b t) c h w")  # B*num_tasks*(T_out), 3, H, W

            target_view_latents: Float[Tensor, "BNt C H W"] = vae.encode(target_view_imgs).latent_dist.sample() * vae.config.scaling_factor
        
            input_depth = batch_data["depth_input"].to(dtype=weight_dtype)  # B,T,3,H,W
            input_depth = rearrange(input_depth, "b t c h w -> (b t) c h w") # B*T,3,H,W
            input_depth_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_depth).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
            prediction_latents = torch.cat([target_view_latents, input_depth_latents], dim=0)
        else:
            batch_size = batch_data["image_target"].shape[0]
            target_rgbs = batch_data["image_target"].to(dtype=weight_dtype)  # B,T,3,H,W
            target_rgbs = rearrange(target_rgbs, "b t c h w -> (b t) c h w") # B*T,3,H,W
            target_rgb_latents = vae.encode(target_rgbs).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
            target_depths = batch_data["depth_target"].to(dtype=weight_dtype)  # B,T,3,H,W
            target_depths = rearrange(target_depths, "b t c h w -> (b t) c h w") # B*T,3,H,W
            target_depth_latents = depth_vae.encode(target_depths).latent_dist.sample() * depth_vae.config.scaling_factor # BT,4,H//8,W//8
            target_rgb_latents = rearrange(target_rgb_latents, "(b t) c h w -> b t c h w", b=batch_size)
            target_depth_latents = rearrange(target_depth_latents, "(b t) c h w -> b t c h w", b=batch_size)
            target_view_latents = torch.cat([target_rgb_latents, target_depth_latents], dim=1)  # B, num_tasks*T, 4, H//8, W//8
            target_view_latents = rearrange(target_view_latents, "b t c h w -> (b t) c h w")  # (B*num_tasks*T, 4, H//8, W//8)
            
            input_depth = batch_data["depth_input"].to(dtype=weight_dtype)  # B,T,3,H,W
            input_depth = rearrange(input_depth, "b t c h w -> (b t) c h w") # B*T,3,H,W
            input_depth_latents = depth_vae.encode(input_depth).latent_dist.sample() * depth_vae.config.scaling_factor  # BT,4,H//8,W//8
            prediction_latents = torch.cat([target_view_latents, input_depth_latents], dim=0)
            
        return prediction_latents
    
    def compose_rgbn_latents(self,
                                batch_data: Dict[str, Any],
                                weight_dtype: torch.dtype,
                                vae: AutoencoderKL,
                                depth_vae: AutoencoderKL = None) -> Float[Tensor, "Bt C H W"]:
        
        target_view_imgs = torch.cat([batch_data["image_target"], 
                                      batch_data["normal_target"]], dim=1).to(dtype=weight_dtype)  # 2B,T,3,H,W
        target_view_imgs = rearrange(target_view_imgs, "b t c h w -> (b t) c h w")  # B*num_tasks*(T_out), 3, H, W

        target_view_latents: Float[Tensor, "BNt C H W"] = vae.encode(target_view_imgs).latent_dist.sample() * vae.config.scaling_factor
        
        input_normal = batch_data["normal_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_normal = rearrange(input_normal, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_normal_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_normal).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
        prediction_latents = torch.cat([target_view_latents, input_normal_latents], dim=0)
        
        return prediction_latents

    def compose_rgbs_latents(self,
                                batch_data: Dict[str, Any],
                                weight_dtype: torch.dtype,
                                vae: AutoencoderKL,
                                depth_vae: AutoencoderKL = None) -> Float[Tensor, "Bt C H W"]:
        
        target_view_imgs = torch.cat([batch_data["image_target"], 
                                      batch_data["semantic_target"]], dim=1).to(dtype=weight_dtype)  # 2B,T,3,H,W
        target_view_imgs = rearrange(target_view_imgs, "b t c h w -> (b t) c h w")  # B*num_tasks*(T_out), 3, H, W

        target_view_latents: Float[Tensor, "BNt C H W"] = vae.encode(target_view_imgs).latent_dist.sample() * vae.config.scaling_factor
        
        input_sem = batch_data["semantic_input"].to(dtype=weight_dtype)  # B,T,3,H,W
        input_sem = rearrange(input_sem, "b t c h w -> (b t) c h w") # B*T,3,H,W
        input_sem_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_sem).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
        prediction_latents = torch.cat([target_view_latents, input_sem_latents], dim=0)
        
        return prediction_latents
    
    def compose_rgb_latents(self,
                            batch_data: Dict[str, Any],
                            weight_dtype: torch.dtype,
                            vae: AutoencoderKL) -> Float[Tensor, "Bt C H W"]:
        target_rgbs = batch_data["image_target"].to(dtype=weight_dtype)  # B,T,3,H,W
        target_rgbs = rearrange(target_rgbs, "b t c h w -> (b t) c h w") # B*T,3,H,W
        target_rgb_latents = vae.encode(target_rgbs).latent_dist.sample() * vae.config.scaling_factor  # BT,4,H//8,W//8
        prediction_latents = target_rgb_latents
            
        return prediction_latents
    
    def compose_prediction_latents(self, 
                                   batch_data: Dict[str, Any], 
                                   weight_dtype: torch.dtype,
                                   vae: AutoencoderKL,
                                   depth_vae: AutoencoderKL = None) -> Float[Tensor, "Bt C H W"]:
        """
        Composes the prediction latents from the input latents
        params:
            batch_data: batch_data data containing input latents
        returns:
            prediction_latents: composed prediction latents
        """

        if self.prediction_types == ["rgb", "depth", "normal", "semantic"]:
            prediction_latents = self.compose_rgbdns_latents(batch_data, weight_dtype, vae, depth_vae)
        elif self.prediction_types == ["rgb", "depth", "normal"]:
            prediction_latents = self.compose_rgbdn_latents(batch_data, weight_dtype, vae)
        elif self.prediction_types == ["rgb", "depth", "semantic"]:
            prediction_latents = self.compose_rgbds_latents(batch_data, weight_dtype, vae)
        elif self.prediction_types == ["rgb", "depth"]:
            prediction_latents = self.compose_rgbd_latents(batch_data, weight_dtype, vae, depth_vae)
        elif self.prediction_types == ["rgb", "normal"]:
            prediction_latents = self.compose_rgbn_latents(batch_data, weight_dtype, vae, depth_vae)
        elif self.prediction_types == ["rgb", "semantic"]:
            prediction_latents = self.compose_rgbs_latents(batch_data, weight_dtype, vae, depth_vae)
        elif self.prediction_types == ["rgb"]:
            prediction_latents = self.compose_rgb_latents(batch_data, weight_dtype, vae)
        else:
            raise ValueError(f"{self.prediction_types} is not supported")
        
        return prediction_latents
    
    def compose_task_embeddings(self,
                                batch_data: Dict[str, Any],
                                weight_dtype: torch.dtype,) -> Float[Tensor, "B C"]:
        
        if self.prediction_types == ["rgb", "depth", "normal", "semantic"]:
            task_embeddings = torch.cat(
                [
                    batch_data["color_task_embeddings"],
                    batch_data["depth_task_embeddings"],
                    batch_data["normal_task_embeddings"],
                    batch_data["semantic_task_embeddings"],
                ],
                dim=1,
            ).to(
                dtype=weight_dtype
            )  # B, 4*num_sample_views, 4
        elif self.prediction_types == ["rgb", "depth", "normal"]:
            task_embeddings = torch.cat(
                [
                    batch_data["color_task_embeddings"],
                    batch_data["depth_task_embeddings"],
                    batch_data["normal_task_embeddings"],
                ],
                dim=1,
            ).to(
                dtype=weight_dtype
            )  # B, 3*num_sample_views, 4
        elif self.prediction_types == ["rgb", "depth", "semantic"]:
            task_embeddings = torch.cat(
                [
                    batch_data["color_task_embeddings"],
                    batch_data["depth_task_embeddings"],
                    batch_data["semantic_task_embeddings"],
                ],
                dim=1,
            ).to(
                dtype=weight_dtype
            )  # B, 3*num_sample_views, 4
        elif self.prediction_types == ["rgb", "depth"]:
            task_embeddings = torch.cat(
                [
                    batch_data["color_task_embeddings"],
                    batch_data["depth_task_embeddings"],
                ],
                dim=1,
            ).to(
                dtype=weight_dtype
            )  # B, 2*num_sample_views, 4
        elif self.prediction_types == ["rgb", "normal"]:
            task_embeddings = torch.cat(
                [
                    batch_data["color_task_embeddings"],
                    batch_data["normal_task_embeddings"],
                ],
                dim=1,
            ).to(dtype=weight_dtype)
        elif self.prediction_types == ["rgb", "semantic"]:
            task_embeddings = torch.cat(
                [
                    batch_data["color_task_embeddings"],
                    batch_data["semantic_task_embeddings"],
                ],
                dim=1,
            ).to(dtype=weight_dtype)
        elif self.prediction_types == ["rgb"]:
            task_embeddings = batch_data["color_task_embeddings"].to(dtype=weight_dtype)
        else:
            raise ValueError(f"prerdiction tasks: {self.prediction_types} is not supported")
        
        if self.use_layout_prior:
            task_embeddings = torch.cat(
                [
                    task_embeddings,
                    batch_data["layout_sem_task_embeddings"],
                    batch_data["layout_depth_task_embeddings"],
                ],
                dim=1,
            ).to(dtype=weight_dtype)
        task_embeddings = rearrange(task_embeddings, "b t c -> (b t) c").contiguous()  # B*num_tasks*num_sample_views, 4

        return task_embeddings

    def compose_view_indices(self,
                             batch_data: Dict[str, Any],
                             ) -> Tuple[Int[Tensor, "Bt"], Int[Tensor, "Bt"], Int[Tensor, "Bt"], Int[Tensor, "Bt"]]:
        bsz = batch_data["input_indices"].shape[0]
        num_tasks = self.num_tasks
        
        # (B, T_in)  (B, T_out)
        input_indices, target_indices = batch_data["input_indices"], batch_data["output_indices"]
        # (B*num_tasks, T_in)
        input_indices = input_indices.repeat_interleave(num_tasks, dim=0)
        # (B*num_tasks, T_out)
        target_indices = target_indices.repeat_interleave(num_tasks, dim=0)

        # convert indices to batch indices
        for batch_idx in range(0, bsz):
            for task_idx in range(0, num_tasks):
                idx = batch_idx * num_tasks + task_idx
                input_indices[idx] = input_indices[idx] + idx * self.num_sample_views
                target_indices[idx] = target_indices[idx] + idx * self.num_sample_views

        if self.prediction_types == ["rgb", "depth", "normal", "semantic"]:
            input_rgb_indices = input_indices[0::6] if self.use_layout_prior else input_indices[0::4]
            input_depth_indices = input_indices[1::6] if self.use_layout_prior else input_indices[1::4]
            input_normal_indices = input_indices[2::6] if self.use_layout_prior else input_indices[2::4]
            input_semantic_indices = input_indices[3::6] if self.use_layout_prior else input_indices[3::4]
            if self.use_layout_prior:
                input_layout_sem_indices = input_indices[4::6]
                input_layout_depth_indices = input_indices[5::6]
                input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
                input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
                
                target_layout_sem_indices = target_indices[4::6]
                target_layout_depth_indices = target_indices[5::6]
                target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
                target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
                
            target_rgb_indices = target_indices[0::6] if self.use_layout_prior else target_indices[0::4]
            target_depth_indices = target_indices[1::6] if self.use_layout_prior else target_indices[1::4]
            target_normal_indices = target_indices[2::6] if self.use_layout_prior else target_indices[2::4]
            target_semantic_indices = target_indices[3::6] if self.use_layout_prior else target_indices[3::4]
                        
            input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
            input_depth_indices = rearrange(input_depth_indices, "B Ni -> (B Ni)")
            input_normal_indices = rearrange(input_normal_indices, "B Ni -> (B Ni)")
            input_semantic_indices = rearrange(input_semantic_indices, "B Ni -> (B Ni)")
            
            pred_target_indices = torch.cat([target_rgb_indices, target_depth_indices, target_normal_indices, target_semantic_indices], dim=1)
            pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
            
            input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
            target_view_indices = rearrange(target_indices, "B No -> (B No)")
            
            prediction_indices = torch.cat([pred_target_indices, 
                                            input_depth_indices, input_normal_indices, input_semantic_indices], dim=0)
            condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                           input_layout_depth_indices, target_layout_depth_indices], dim=0) if self.use_layout_prior else input_rgb_indices
            
        elif self.prediction_types == ["rgb", "depth", "normal"]:
            input_rgb_indices = input_indices[0::5] if self.use_layout_prior else input_indices[0::3] 
            input_depth_indices = input_indices[1::5] if self.use_layout_prior else input_indices[1::3]
            input_normal_indices = input_indices[2::5] if self.use_layout_prior else input_indices[2::3]
            if self.use_layout_prior:
                input_layout_sem_indices = input_indices[3::5]
                input_layout_depth_indices = input_indices[4::5]
                input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
                input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
                
                target_layout_sem_indices = target_indices[3::5]
                target_layout_depth_indices = target_indices[4::5]
                target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
                target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
            
            target_rgb_indices = target_indices[0::5] if self.use_layout_prior else target_indices[0::3]
            target_depth_indices = target_indices[1::5] if self.use_layout_prior else target_indices[1::3]
            target_normal_indices = target_indices[2::5] if self.use_layout_prior else target_indices[2::3]
            
            input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
            input_depth_indices = rearrange(input_depth_indices, "B Ni -> (B Ni)")
            input_normal_indices = rearrange(input_normal_indices, "B Ni -> (B Ni)")
            
            pred_target_indices = torch.cat([target_rgb_indices, target_depth_indices, target_normal_indices], dim=1)
            pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
            
            input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
            target_view_indices = rearrange(target_indices, "B No -> (B No)")
            
            prediction_indices = torch.cat([pred_target_indices,
                                            input_depth_indices, input_normal_indices], dim=0)
            condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices, 
                                           input_layout_depth_indices, target_layout_depth_indices], dim=0) if self.use_layout_prior else input_rgb_indices
             
        elif self.prediction_types == ["rgb", "depth", "semantic"]:
            input_rgb_indices = input_indices[0::5] if self.use_layout_prior else input_indices[0::3] 
            input_depth_indices = input_indices[1::5] if self.use_layout_prior else input_indices[1::3]
            input_sem_indices = input_indices[2::5] if self.use_layout_prior else input_indices[2::3]
            if self.use_layout_prior:
                input_layout_sem_indices = input_indices[3::5]
                input_layout_depth_indices = input_indices[4::5]
                input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
                input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
                
                target_layout_sem_indices = target_indices[3::5]
                target_layout_depth_indices = target_indices[4::5]
                target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
                target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
            
            target_rgb_indices = target_indices[0::5] if self.use_layout_prior else target_indices[0::3]
            target_depth_indices = target_indices[1::5] if self.use_layout_prior else target_indices[1::3]
            target_sem_indices = target_indices[2::5] if self.use_layout_prior else target_indices[2::3]
            
            input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
            input_depth_indices = rearrange(input_depth_indices, "B Ni -> (B Ni)")
            input_sem_indices = rearrange(input_sem_indices, "B Ni -> (B Ni)")
            
            pred_target_indices = torch.cat([target_rgb_indices, target_depth_indices, target_sem_indices], dim=1)
            pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
            
            input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
            target_view_indices = rearrange(target_indices, "B No -> (B No)")
            
            prediction_indices = torch.cat([pred_target_indices,
                                            input_depth_indices, input_sem_indices], dim=0)
            condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices, 
                                           input_layout_depth_indices, target_layout_depth_indices], dim=0) if self.use_layout_prior else input_rgb_indices
             
        elif self.prediction_types == ["rgb", "normal"]:
            input_rgb_indices = input_indices[0::4] if self.use_layout_prior else input_indices[0::2]
            input_normal_indices = input_indices[1::4] if self.use_layout_prior else input_indices[1::2]
            if self.use_layout_prior:
                input_layout_sem_indices = input_indices[2::4]
                input_layout_depth_indices = input_indices[3::4]
                input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
                input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
                
                target_layout_sem_indices = target_indices[2::4]
                target_layout_depth_indices = target_indices[3::4]
                target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
                target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
                
            target_rgb_indices = target_indices[0::4] if self.use_layout_prior else target_indices[0::2]
            target_normal_indices = target_indices[1::4] if self.use_layout_prior else target_indices[1::2]
            
            input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
            input_normal_indices = rearrange(input_normal_indices, "B Ni -> (B Ni)")
            
            pred_target_indices = torch.cat([target_rgb_indices, target_normal_indices], dim=1)
            pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
            
            input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
            target_view_indices = rearrange(target_indices, "B No -> (B No)")
            
            prediction_indices = torch.cat([pred_target_indices, input_normal_indices], dim=0)
            condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                           input_layout_depth_indices, target_layout_depth_indices], dim=0) if self.use_layout_prior else input_rgb_indices
            # logger.info(f"input_normal_indices: {input_normal_indices}")
        elif self.prediction_types == ["rgb", "depth"]:
            input_rgb_indices = input_indices[0::4] if self.use_layout_prior else input_indices[0::2] 
            input_depth_indices = input_indices[1::4] if self.use_layout_prior else input_indices[1::2]
            if self.use_layout_prior:
                input_layout_sem_indices = input_indices[2::4]
                input_layout_depth_indices = input_indices[3::4]
                input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
                input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
                
                target_layout_sem_indices = target_indices[2::4]
                target_layout_depth_indices = target_indices[3::4]
                target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
                target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
                
            target_rgb_indices = target_indices[0::4] if self.use_layout_prior else target_indices[0::2]
            target_depth_indices = target_indices[1::4] if self.use_layout_prior else target_indices[1::2]
            
            input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
            input_depth_indices = rearrange(input_depth_indices, "B Ni -> (B Ni)")
            
            pred_target_indices = torch.cat([target_rgb_indices, target_depth_indices], dim=1)
            pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
            
            input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
            target_view_indices = rearrange(target_indices, "B No -> (B No)")
            
            prediction_indices = torch.cat([pred_target_indices, input_depth_indices], dim=0)
            condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                           input_layout_depth_indices, target_layout_depth_indices], dim=0) if self.use_layout_prior else input_rgb_indices
        elif self.prediction_types == ["rgb", "semantic"]:
            input_rgb_indices = input_indices[0::4] if self.use_layout_prior else input_indices[0::2] 
            input_sem_indices = input_indices[1::4] if self.use_layout_prior else input_indices[1::2]
            if self.use_layout_prior:
                input_layout_sem_indices = input_indices[2::4]
                input_layout_depth_indices = input_indices[3::4]
                input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
                input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
                
                target_layout_sem_indices = target_indices[2::4]
                target_layout_depth_indices = target_indices[3::4]
                target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
                target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
                
            target_rgb_indices = target_indices[0::4] if self.use_layout_prior else target_indices[0::2]
            target_sem_indices = target_indices[1::4] if self.use_layout_prior else target_indices[1::2]
            
            input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
            input_sem_indices = rearrange(input_sem_indices, "B Ni -> (B Ni)")
            
            pred_target_indices = torch.cat([target_rgb_indices, target_sem_indices], dim=1)
            pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
            
            input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
            target_view_indices = rearrange(target_indices, "B No -> (B No)")
            
            prediction_indices = torch.cat([pred_target_indices, input_sem_indices], dim=0)
            condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                           input_layout_depth_indices, target_layout_depth_indices], dim=0) if self.use_layout_prior else input_rgb_indices
        elif self.prediction_types == ["rgb"]:
            input_rgb_indices = input_indices[0::3] if self.use_layout_prior else input_indices
            if self.use_layout_prior:
                input_layout_sem_indices = input_indices[1::3]
                input_layout_depth_indices = input_indices[2::3]
                input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
                input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
                
                target_layout_sem_indices = target_indices[1::3]
                target_layout_depth_indices = target_indices[2::3]
                target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
                target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
            
            target_rgb_indices = target_indices[0::3] if self.use_layout_prior else target_indices
            
            input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
            target_rgb_indices = rearrange(target_rgb_indices, "B No -> (B No)")
            
            input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
            target_view_indices = rearrange(target_indices, "B No -> (B No)")
            
            prediction_indices = target_rgb_indices
            condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                           input_layout_depth_indices, target_layout_depth_indices], dim=0) if self.use_layout_prior else input_rgb_indices
        else:
            raise ValueError(f"{self.prediction_types} is not supported")
        # logger.info(f"input_rgb_indices: {input_rgb_indices}")
        # logger.info(f"target_view_indices: {target_view_indices}")
        # logger.info(f"prediction_indices: {prediction_indices}")
        
        return input_rgb_indices, condition_indices, input_view_indices, target_view_indices, prediction_indices
                
    def forward(
        self,
        input_rgb_latents: Float[Tensor, "Bt C H W"],
        condition_latents: Float[Tensor, "Bt C H W"],
        noisy_latents: Float[Tensor, "Bt C H W"],
        warpped_target_rgb_latents: Float[Tensor, "Bt 4 H W"],
        input_view_rays: Float[Tensor, "B N 6 H W"],
        target_view_rays: Float[Tensor, "B N 6 H W"],
        bs_text_embeds: Float[Tensor, "B L C"],
        timesteps_all: Float[Tensor, "BNt"],
        task_embeddings: Float[Tensor, "B C"],
        input_rgb_indices: Int[Tensor, "Bt"],
        condition_indices: Int[Tensor, "Bt"],
        input_view_indices: Int[Tensor, "Bt"],
        target_view_indices: Int[Tensor, "Bt"],
        prediction_indices: Int[Tensor, "Bt"],
        ray_drop_masks: Float[Tensor, "B 1"],
        cond_image_masks: Float[Tensor, "Bt 1 H W"] = None,
        bs_text_attention_masks: Float[Tensor, "B L"] = None,
    ):
        bsz = input_view_rays.shape[0]
        weight_type = input_rgb_latents.dtype

        input_ray_embeds = self.encode_cam_rays(input_view_rays, ray_drop_masks).to(weight_type)  # (B*num_tasks*Nt, 16, H//8, W//8)
        target_ray_embeds = self.encode_cam_rays(target_view_rays, ray_drop_masks).to(weight_type)  # (B*num_tasks*Nt, 16, H//8, W//8)
        
        # (B*num_tasks*num_views, 21, Hl, Wl), 21 = 4(latents) + 1(mask) + 16(rays)
        model_inputs = torch.zeros(
            (
                bsz * self.num_tasks * self.num_sample_views,
                self.unet_in_channels,
                input_rgb_latents.shape[-2],
                input_rgb_latents.shape[-1],
            ),
            device=input_rgb_latents.device,
            dtype=weight_type,
        )
        # latents
        model_inputs[condition_indices, :4, ...] = condition_latents
        model_inputs[prediction_indices, :4, ...] = noisy_latents
        
        # ray mebeddings
        end_ray_chann = 4 + self.ray_channel
        model_inputs[input_view_indices, 4:end_ray_chann, ...] = input_ray_embeds
        model_inputs[target_view_indices, 4:end_ray_chann, ...] = target_ray_embeds
        
        # binary masks
        mask_channel = 1
        model_inputs[condition_indices, end_ray_chann:end_ray_chann+mask_channel, ...] = 1.0
        model_inputs[input_rgb_indices, end_ray_chann:end_ray_chann+mask_channel, ...] = cond_image_masks
        model_inputs[prediction_indices, end_ray_chann:end_ray_chann+mask_channel, ...] = 0.0
        if warpped_target_rgb_latents is not None:
            # warpped rgbs latents
            input_rgb_latents_tmp = rearrange(input_rgb_latents, "(B Nt) C H W -> B Nt C H W", B=bsz)  # B, T_in, 4, H, W
            input_rgb_latents_tmp = input_rgb_latents_tmp.repeat(1, self.num_tasks, 1, 1, 1)  # B, num_tasks*T_in, 4, H, W
            input_rgb_latents_tmp = rearrange(input_rgb_latents_tmp, "B Nt C H W -> (B Nt) C H W")  # (B*num_tasks*T_in, 4, H, W)
            warpped_target_rgb_latents_tmp = rearrange(warpped_target_rgb_latents, "(B Nt) C H W -> B Nt C H W", B=bsz)  # B, T_out, 4, H, W
            warpped_target_rgb_latents_tmp = warpped_target_rgb_latents_tmp.repeat(1, self.num_tasks, 1, 1, 1)  # B, num_taks*T_out, 4, H, W
            warpped_target_rgb_latents_tmp = rearrange(warpped_target_rgb_latents_tmp, "B Nt C H W -> (B Nt) C H W") # (B*num_tasks, T_out, 4, H, W)

            model_inputs[input_view_indices, end_ray_chann+mask_channel:end_ray_chann+mask_channel+4, ...] = input_rgb_latents_tmp
            model_inputs[target_view_indices, end_ray_chann+mask_channel:end_ray_chann+mask_channel+4, ...] = warpped_target_rgb_latents_tmp

        # Predict the noise residual
        model_pred = self.unet(
            model_inputs,
            encoder_hidden_states=bs_text_embeds,  # (b*num_tasks) l 1024
            encoder_attention_mask=bs_text_attention_masks,
            timestep=timesteps_all,
            class_labels=task_embeddings,
        ).sample

        # only return the predicted noise for the predicted views
        noise_pred = model_pred[prediction_indices]
        return noise_pred, model_inputs

