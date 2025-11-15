import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from typing import *
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.data_loader import DataLoaderShard

import os
import os.path as osp
import argparse
import logging
import math
from collections import defaultdict
from packaging import version
import gc
import shutil

from tqdm import tqdm
import wandb
import numpy as np
from skimage.metrics import structural_similarity as calculate_ssim
from lpips import LPIPS

import torch
import torchvision
import torch.nn.functional as tF
from einops import rearrange, repeat
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from diffusers import DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, AutoencoderKL, AutoencoderTiny
from diffusers.training_utils import compute_snr
from transformers import CLIPTokenizer, CLIPTextModel

from src.options import opt_dict, Options
from src.data import MixDataset
from src.models import get_optimizer, get_lr_scheduler
from src.models.cat3d_adapter import CAT3DAdaptor
from src.models.pose_adapter import RayMapEncoder, RayMapEncoderConfig

import src.utils.util as util
from src.utils.typing import *
from src.utils.pcl_ops import save_input_output_pointcloud, save_input_output_pointcloud_with_sem, cross_viewpoint_rendering, cross_viewpoint_rendering_pt3d
from src.utils.vis_util import apply_depth_to_colormap, save_color_depth_image
from src.utils.misc import print_memory, worker_init_fn, enable_flash_attn_if_avail

from diffusers_spatialgen import MyEMAModel, SpatialGenDiffusionPipeline
from diffusers_spatialgen import UNetMVMM2DConditionModel

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_autotune"

logger = get_accelerate_logger(__name__, log_level="INFO")

def compose_view_indices(batch:Dict[str, Tensor], opt: Options):
    
    input_indices, target_indices = batch["input_indices"][0:1], batch["output_indices"][0:1]
            
    num_sample_views = opt.num_views
    num_tasks = len(opt.prediction_types) + 2 if opt.use_layout_prior else len(opt.prediction_types) 
    # (2*B, T_in), (2*B, T_out)
    input_indices, target_indices = input_indices.repeat(num_tasks, 1), target_indices.repeat(num_tasks, 1)
    # convert indices to batch indices
    for batch_idx in range(1 * num_tasks):
        input_indices[batch_idx] = input_indices[batch_idx] + batch_idx * num_sample_views
        target_indices[batch_idx] = target_indices[batch_idx] + batch_idx * num_sample_views

    if opt.prediction_types == ["rgb", "depth", "normal", "semantic"]:
        input_rgb_indices = input_indices[0::6] if opt.use_layout_prior else input_indices[0::4]
        input_depth_indices = input_indices[1::6] if opt.use_layout_prior else input_indices[1::4]
        input_normal_indices = input_indices[2::6] if opt.use_layout_prior else input_indices[2::4]
        input_semantic_indices = input_indices[3::6] if opt.use_layout_prior else input_indices[3::4]
        if opt.use_layout_prior:
            input_layout_sem_indices = input_indices[4::6]
            input_layout_depth_indices = input_indices[5::6]
            input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
            input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
            
            target_layout_sem_indices = target_indices[4::6]
            target_layout_depth_indices = target_indices[5::6]
            target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
            target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
            
        target_rgb_indices = target_indices[0::6] if opt.use_layout_prior else target_indices[0::4]
        target_depth_indices = target_indices[1::6] if opt.use_layout_prior else target_indices[1::4]
        target_normal_indices = target_indices[2::6] if opt.use_layout_prior else target_indices[2::4]
        target_semantic_indices = target_indices[3::6] if opt.use_layout_prior else target_indices[3::4]
                    
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
                                        input_layout_depth_indices, target_layout_depth_indices], dim=0) if opt.use_layout_prior else input_rgb_indices
        
    elif opt.prediction_types == ["rgb", "depth", "semantic"]:
        input_rgb_indices = input_indices[0::5] if opt.use_layout_prior else input_indices[0::3] 
        input_depth_indices = input_indices[1::5] if opt.use_layout_prior else input_indices[1::3]
        input_sem_indices = input_indices[2::5] if opt.use_layout_prior else input_indices[2::3]
        if opt.use_layout_prior:
            input_layout_sem_indices = input_indices[3::5]
            input_layout_depth_indices = input_indices[4::5]
            input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
            input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
            
            target_layout_sem_indices = target_indices[3::5]
            target_layout_depth_indices = target_indices[4::5]
            target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
            target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
        
        target_rgb_indices = target_indices[0::5] if opt.use_layout_prior else target_indices[0::3]
        target_depth_indices = target_indices[1::5] if opt.use_layout_prior else target_indices[1::3]
        target_sem_indices = target_indices[2::5] if opt.use_layout_prior else target_indices[2::3]
        
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
                                        input_layout_depth_indices, target_layout_depth_indices], dim=0) if opt.use_layout_prior else input_rgb_indices
            
    elif opt.prediction_types == ["rgb", "normal"]:
        input_rgb_indices = input_indices[0::4] if opt.use_layout_prior else input_indices[0::2]
        input_normal_indices = input_indices[1::4] if opt.use_layout_prior else input_indices[1::2]
        if opt.use_layout_prior:
            input_layout_sem_indices = input_indices[2::4]
            input_layout_depth_indices = input_indices[3::4]
            input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
            input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
            
            target_layout_sem_indices = target_indices[2::4]
            target_layout_depth_indices = target_indices[3::4]
            target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
            target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
            
        target_rgb_indices = target_indices[0::4] if opt.use_layout_prior else target_indices[0::2]
        target_normal_indices = target_indices[1::4] if opt.use_layout_prior else target_indices[1::2]
        
        input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
        input_normal_indices = rearrange(input_normal_indices, "B Ni -> (B Ni)")
        
        pred_target_indices = torch.cat([target_rgb_indices, target_normal_indices], dim=1)
        pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
        
        input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
        target_view_indices = rearrange(target_indices, "B No -> (B No)")
        
        prediction_indices = torch.cat([pred_target_indices, input_normal_indices], dim=0)
        condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                        input_layout_depth_indices, target_layout_depth_indices], dim=0) if opt.use_layout_prior else input_rgb_indices

    elif opt.prediction_types == ["rgb", "depth"]:
        input_rgb_indices = input_indices[0::4] if opt.use_layout_prior else input_indices[0::2] 
        input_depth_indices = input_indices[1::4] if opt.use_layout_prior else input_indices[1::2]
        if opt.use_layout_prior:
            input_layout_sem_indices = input_indices[2::4]
            input_layout_depth_indices = input_indices[3::4]
            input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
            input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
            
            target_layout_sem_indices = target_indices[2::4]
            target_layout_depth_indices = target_indices[3::4]
            target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
            target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
            
        target_rgb_indices = target_indices[0::4] if opt.use_layout_prior else target_indices[0::2]
        target_depth_indices = target_indices[1::4] if opt.use_layout_prior else target_indices[1::2]
        
        input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
        input_depth_indices = rearrange(input_depth_indices, "B Ni -> (B Ni)")
        
        pred_target_indices = torch.cat([target_rgb_indices, target_depth_indices], dim=1)
        pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
        
        input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
        target_view_indices = rearrange(target_indices, "B No -> (B No)")
        
        prediction_indices = torch.cat([pred_target_indices, input_depth_indices], dim=0)
        condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                        input_layout_depth_indices, target_layout_depth_indices], dim=0) if opt.use_layout_prior else input_rgb_indices
    elif opt.prediction_types == ["rgb", "semantic"]:
        input_rgb_indices = input_indices[0::4] if opt.use_layout_prior else input_indices[0::2] 
        input_sem_indices = input_indices[1::4] if opt.use_layout_prior else input_indices[1::2]
        if opt.use_layout_prior:
            input_layout_sem_indices = input_indices[2::4]
            input_layout_depth_indices = input_indices[3::4]
            input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
            input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
            
            target_layout_sem_indices = target_indices[2::4]
            target_layout_depth_indices = target_indices[3::4]
            target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
            target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
            
        target_rgb_indices = target_indices[0::4] if opt.use_layout_prior else target_indices[0::2]
        target_sem_indices = target_indices[1::4] if opt.use_layout_prior else target_indices[1::2]
        
        input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
        input_sem_indices = rearrange(input_sem_indices, "B Ni -> (B Ni)")
        
        pred_target_indices = torch.cat([target_rgb_indices, target_sem_indices], dim=1)
        pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")
        
        input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
        target_view_indices = rearrange(target_indices, "B No -> (B No)")
        
        prediction_indices = torch.cat([pred_target_indices, input_sem_indices], dim=0)
        condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                        input_layout_depth_indices, target_layout_depth_indices], dim=0) if opt.use_layout_prior else input_rgb_indices
    elif opt.prediction_types == ["rgb"]:
        input_rgb_indices = input_indices[0::3] if opt.use_layout_prior else input_indices
        if opt.use_layout_prior:
            input_layout_sem_indices = input_indices[1::3]
            input_layout_depth_indices = input_indices[2::3]
            input_layout_sem_indices = rearrange(input_layout_sem_indices, "B Ni -> (B Ni)")
            input_layout_depth_indices = rearrange(input_layout_depth_indices, "B Ni -> (B Ni)")
            
            target_layout_sem_indices = target_indices[1::3]
            target_layout_depth_indices = target_indices[2::3]
            target_layout_sem_indices = rearrange(target_layout_sem_indices, "B No -> (B No)")
            target_layout_depth_indices = rearrange(target_layout_depth_indices, "B No -> (B No)")
        
        target_rgb_indices = target_indices[0::3] if opt.use_layout_prior else target_indices
        
        input_rgb_indices = rearrange(input_rgb_indices, "B Ni -> (B Ni)")
        target_rgb_indices = rearrange(target_rgb_indices, "B No -> (B No)")
        
        input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
        target_view_indices = rearrange(target_indices, "B No -> (B No)")
        
        prediction_indices = target_rgb_indices
        condition_indices = torch.cat([input_rgb_indices, input_layout_sem_indices, target_layout_sem_indices,
                                        input_layout_depth_indices, target_layout_depth_indices], dim=0) if opt.use_layout_prior else input_rgb_indices
    else:
        raise ValueError(f"{opt.prediction_types} is not supported")
    
    return input_rgb_indices, condition_indices, input_view_indices, target_view_indices, prediction_indices
            
def logging_mv_mm_images(opt: Options, batch:Dict[str, Tensor], pred_images:np.ndarray, gt_images:Tensor,
                         input_image:Tensor,
                         num_in_views: int = 1, num_out_views: int = 1,
                         output_folder: str = './',):
    
    room_uid = batch["room_uid"][0].replace("/", "_")

    # save inputs, preditions, and gt
    if ["rgb"] == opt.prediction_types:
        num_target_rgbs = num_out_views
        pred_rgbs = torch.from_numpy(pred_images[:num_target_rgbs, :, :, :]).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        gt_rgbs = (gt_images[0:num_target_rgbs] / 2 + 0.5).clamp(0, 1).cpu()
        inputs = torchvision.utils.make_grid(input_image / 2. + 0.5, nrow=1)
        torchvision.utils.save_image(inputs, f"{output_folder}/input_rgbs.png")
        predictions = torchvision.utils.make_grid(pred_rgbs, nrow=1)
        torchvision.utils.save_image(predictions, f"{output_folder}/pred_tar_rgbs.png")
        gts = torchvision.utils.make_grid(gt_rgbs, nrow=1)
        torchvision.utils.save_image(gts, f"{output_folder}/gt_tar_rgbs.png")
    elif ["rgb", "semantic"] == opt.prediction_types:
        num_target_rgbs, num_target_semantics, num_input_semantics = num_out_views, num_out_views, num_in_views
        rgb_idx = num_target_rgbs
        semantic_idx = num_target_rgbs + num_target_semantics
        in_semantic_idx = semantic_idx + num_input_semantics
        pred_images = torch.from_numpy(pred_images).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        fake_tar_rgbs = pred_images[:rgb_idx]  # (BxT)x3xhxw
        fake_tar_sems = pred_images[rgb_idx:semantic_idx]  # (BxT)x3xhxw
        fake_in_sems = pred_images[semantic_idx:in_semantic_idx]  # (BxT)x3xhxw
        
        inputs = torchvision.utils.make_grid(input_image / 2. + 0.5, nrow=1)
        torchvision.utils.save_image(inputs, f"{output_folder}/input_rgbs.png")
        fake_tar_rgbs = torchvision.utils.make_grid(fake_tar_rgbs, nrow=1)
        torchvision.utils.save_image(fake_tar_rgbs, f"{output_folder}/pred_tar_rgbs.png")
        pred_sems = torchvision.utils.make_grid(fake_tar_sems, nrow=1)
        torchvision.utils.save_image(pred_sems, f"{output_folder}/pred_tar_semantics.png")
        pred_in_sems = torchvision.utils.make_grid(fake_in_sems, nrow=1)
        torchvision.utils.save_image(pred_in_sems, f"{output_folder}/pred_in_semantics.png")
        gts = torchvision.utils.make_grid(gt_rgbs, nrow=1)
        torchvision.utils.save_image(gts, f"{output_folder}/gt_tar_rgbs.png")
        gt_sems = (gt_images[rgb_idx:semantic_idx] * 0.5 + 0.5).clamp(0, 1).cpu()
        gt_sems = torchvision.utils.make_grid(gt_sems, nrow=1)
        torchvision.utils.save_image(gt_sems, f"{output_folder}/gt_tar_semantics.png")
        gt_in_sems = (gt_images[semantic_idx:in_semantic_idx] * 0.5 + 0.5).clamp(0, 1).cpu()
        gt_in_sems = torchvision.utils.make_grid(gt_in_sems, nrow=1)
        torchvision.utils.save_image(gt_in_sems, f"{output_folder}/gt_in_semantics.png")
    # recover the decoded depth into point cloud
    elif ["rgb", "depth"] == opt.prediction_types:
        num_target_rgbs, num_target_depths, num_input_depths = num_out_views, num_out_views, num_in_views
        rgb_idx = num_target_rgbs
        depth_idx = num_target_rgbs + num_target_depths
        in_depth_idx = depth_idx + num_input_depths
        fake_tar_rgbs = torch.from_numpy(pred_images[:rgb_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        if opt.use_scene_coord_map:
            # actual scene coord map
            fake_tar_depths = torch.from_numpy(pred_images[rgb_idx:depth_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
            fake_in_depths = torch.from_numpy(pred_images[depth_idx:in_depth_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        else:
            fake_tar_depths = torch.from_numpy(pred_images[rgb_idx:depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x1xhxw
            fake_in_depths = torch.from_numpy(pred_images[depth_idx:in_depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x1xhxw
        gt_tar_rgbs = rearrange(batch["image_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,3,h,w
        if opt.use_scene_coord_map:
            gt_tar_depths = rearrange(batch["depth_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,3,h,w
            gt_in_depths = rearrange(batch["depth_input"][0:1], "b t c h w -> (b t) c h w", t=num_in_views).cpu()  # B,3,h,w
        else:
            gt_tar_depths = rearrange(batch["depth_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,1,h,w
            gt_in_depths = rearrange(batch["depth_input"][0:1], "b t c h w -> (b t) c h w", t=num_in_views).cpu()[:, 0:1, :, :]  # B,1,h,w

        pose_in = batch["pose_in"][0:1]  # B,T_in,4,4
        pose_out = batch["pose_out"][0:1]  # B,T_out,4,4
        min_depth = batch["depth_min"][0:1].cpu()  # B,1
        max_depth = batch["depth_max"][0:1].cpu()
        scene_scale = batch["scene_scale"][0:1].cpu()
        intrinsic_mat = batch["intrinsic"][0].cpu().numpy()
        print("validating RGB-D task on room {}".format(room_uid))

        save_input_output_pointcloud(
            input_images=input_image.float(),
            input_depths=fake_in_depths.float(),
            output_images=fake_tar_rgbs.float(),
            output_depths=fake_tar_depths.float(),
            poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=num_in_views),
            poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=num_out_views),
            min_depth=min_depth.float(),
            max_depth=max_depth.float(),
            scene_scale=scene_scale.float(),
            use_metric_depth=opt.use_metric_depth,
            use_scene_coord_map=opt.use_scene_coord_map,
            output_folder=output_folder,
            intrinsic_mat=intrinsic_mat,
            is_gt=False,
        )
        save_input_output_pointcloud(
            input_images=input_image.float(),
            input_depths=gt_in_depths.float(),
            output_images=gt_tar_rgbs.float(),
            output_depths=gt_tar_depths.float(),
            poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=num_in_views),
            poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=num_out_views),
            min_depth=min_depth.float(),
            max_depth=max_depth.float(),
            scene_scale=scene_scale.float(),
            use_metric_depth=opt.use_metric_depth,
            use_scene_coord_map=opt.use_scene_coord_map,
            output_folder=output_folder,
            intrinsic_mat=intrinsic_mat,
            is_gt=True,
        )
            
        if not opt.use_scene_coord_map:
            # convert depth into colorful depth
            fake_tar_depths = pred_images[rgb_idx:depth_idx, :, :, 0]
            fake_tar_depths = save_color_depth_image(fake_tar_depths, output_path=f"{output_folder}/pred_tar_depths.png")  # BxT_out,H,W,3
            pred_images[rgb_idx:depth_idx, :, :, :] = fake_tar_depths
            
            fake_in_depths = pred_images[depth_idx:in_depth_idx, :, :, 0]
            fake_in_depths = save_color_depth_image(fake_in_depths, output_path=f"{output_folder}/pred_in_depths.png")  # BxT_in,H,W,3
            pred_images[depth_idx:in_depth_idx, :, :, :] = fake_in_depths

            gt_tar_depths = ((gt_tar_depths + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].float().numpy()
            gt_tar_depths = save_color_depth_image(gt_tar_depths, output_path=f"{output_folder}/gt_tar_depths.png")  # BxT_out,H,W,3
            gt_tar_depths = gt_tar_depths * 2 - 1.0
            
            gt_in_depths = ((gt_in_depths + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].float().numpy()
            gt_in_depths = save_color_depth_image(gt_in_depths, output_path=f"{output_folder}/gt_in_depths.png")  # BxT_in,H,W,3
            gt_in_depths = gt_in_depths * 2 - 1.0
            
            gt_images[rgb_idx:depth_idx, :, :, :] = torch.from_numpy(gt_tar_depths).to(input_image).permute(0, 3, 1, 2)
            gt_images[depth_idx:in_depth_idx, :, :, :] = torch.from_numpy(gt_in_depths).to(input_image).permute(0, 3, 1, 2)
        else:
            torchvision.utils.save_image(torchvision.utils.make_grid(fake_tar_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/pred_tar_depths.png")
            torchvision.utils.save_image(torchvision.utils.make_grid(fake_in_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/pred_in_depths.png")
            torchvision.utils.save_image(torchvision.utils.make_grid(gt_tar_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/gt_tar_depths.png")
            torchvision.utils.save_image(torchvision.utils.make_grid(gt_in_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/gt_in_depths.png")

    elif ["rgb", "depth", "semantic"] == opt.prediction_types:
        num_target_rgbs, num_target_depths, num_target_sems, num_input_depths, num_input_sems = num_out_views, num_out_views, num_out_views, num_in_views, num_in_views
        rgb_idx = num_target_rgbs
        depth_idx = num_target_rgbs + num_target_depths
        sem_idx = num_target_rgbs + num_target_depths + num_target_sems
        in_depth_idx = sem_idx + num_input_depths
        in_sem_idx = in_depth_idx + num_input_sems
        fake_tar_rgbs = torch.from_numpy(pred_images[:rgb_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        if opt.use_scene_coord_map:
            # actual scene coord map
            fake_tar_depths = torch.from_numpy(pred_images[rgb_idx:depth_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
            fake_in_depths = torch.from_numpy(pred_images[sem_idx:in_depth_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        else:
            fake_tar_depths = torch.from_numpy(pred_images[rgb_idx:depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x1xhxw
            fake_in_depths = torch.from_numpy(pred_images[sem_idx:in_depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x1xhxw
        fake_tar_sems = torch.from_numpy(pred_images[depth_idx:sem_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        fake_in_sems = torch.from_numpy(pred_images[in_depth_idx:in_sem_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        
        gt_tar_rgbs = rearrange(batch["image_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,3,h,w
        if opt.use_scene_coord_map:
            gt_tar_depths = rearrange(batch["depth_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,3,h,w
            gt_in_depths = rearrange(batch["depth_input"][0:1], "b t c h w -> (b t) c h w", t=num_in_views).cpu()  # B,3,h,w
        else:
            gt_tar_depths = rearrange(batch["depth_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,1,h,w
            gt_in_depths = rearrange(batch["depth_input"][0:1], "b t c h w -> (b t) c h w", t=num_in_views).cpu()[:, 0:1, :, :]  # B,1,h,w
        gt_tar_sems = rearrange(batch["semantic_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,3,h,w
        gt_in_sems = rearrange(batch["semantic_input"][0:1], "b t c h w -> (b t) c h w", t=num_in_views).cpu()  # B,3,h,w
        
        pose_in = batch["pose_in"][0:1]  # B,T_in,4,4
        pose_out = batch["pose_out"][0:1]  # B,T_out,4,4
        min_depth = batch["depth_min"][0:1].cpu()  # B,1
        max_depth = batch["depth_max"][0:1].cpu()
        scene_scale = batch["scene_scale"][0:1].cpu()
        intrinsic_mat = batch["intrinsic"][0].cpu().numpy()
        print("validating RGB-D-Sem task on room {}".format(room_uid))

        save_input_output_pointcloud(
            input_images=input_image.float(),
            input_depths=fake_in_depths.float(),
            output_images=fake_tar_rgbs.float(),
            output_depths=fake_tar_depths.float(),
            poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=num_in_views),
            poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=num_out_views),
            min_depth=min_depth.float(),
            max_depth=max_depth.float(),
            scene_scale=scene_scale.float(),
            use_metric_depth=opt.use_metric_depth,
            use_scene_coord_map=opt.use_scene_coord_map,
            output_folder=output_folder,
            intrinsic_mat=intrinsic_mat,
            is_gt=False,
        )
        save_input_output_pointcloud(
            input_images=input_image.float(),
            input_depths=gt_in_depths.float(),
            output_images=gt_tar_rgbs.float(),
            output_depths=gt_tar_depths.float(),
            poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=num_in_views),
            poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=num_out_views),
            min_depth=min_depth.float(),
            max_depth=max_depth.float(),
            scene_scale=scene_scale.float(),
            use_metric_depth=opt.use_metric_depth,
            use_scene_coord_map=opt.use_scene_coord_map,
            output_folder=output_folder,
            intrinsic_mat=intrinsic_mat,
            is_gt=True,
        )
        
        # save pred semantic images
        pred_tar_semantics = torchvision.utils.make_grid(fake_tar_sems*0.5+0.5, nrow=1)
        torchvision.utils.save_image(pred_tar_semantics, f"{output_folder}/pred_tar_semantics.png")
        pred_in_semantics = torchvision.utils.make_grid(fake_in_sems*0.5+0.5, nrow=1)
        torchvision.utils.save_image(pred_in_semantics, f"{output_folder}/pred_in_semantics.png")
        gt_tar_semantics = torchvision.utils.make_grid(gt_tar_sems*0.5+0.5, nrow=1)
        torchvision.utils.save_image(gt_tar_semantics, f"{output_folder}/gt_tar_semantics.png")
        gt_in_sems = torchvision.utils.make_grid(gt_in_sems*0.5+0.5, nrow=1)
        torchvision.utils.save_image(gt_in_sems, f"{output_folder}/gt_in_semantics.png")
            
        if not opt.use_scene_coord_map:
            # convert depth into colorful depth
            fake_tar_depths = pred_images[rgb_idx:depth_idx, :, :, 0]
            fake_tar_depths = save_color_depth_image(fake_tar_depths, output_path=f"{output_folder}/pred_tar_depths.png")  # BxT_out,H,W,3
            pred_images[rgb_idx:depth_idx, :, :, :] = fake_tar_depths
            
            fake_in_depths = pred_images[sem_idx:in_depth_idx, :, :, 0]
            fake_in_depths = save_color_depth_image(fake_in_depths, output_path=f"{output_folder}/pred_in_depths.png")  # BxT_in,H,W,3
            pred_images[depth_idx:in_depth_idx, :, :, :] = fake_in_depths

            gt_tar_depths = ((gt_tar_depths + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].float().numpy()
            gt_tar_depths = save_color_depth_image(gt_tar_depths, output_path=f"{output_folder}/gt_tar_depths.png")  # BxT_out,H,W,3
            gt_tar_depths = gt_tar_depths * 2 - 1.0
            
            gt_in_depths = ((gt_in_depths + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].float().numpy()
            gt_in_depths = save_color_depth_image(gt_in_depths, output_path=f"{output_folder}/gt_in_depths.png")  # BxT_in,H,W,3
            gt_in_depths = gt_in_depths * 2 - 1.0
            
            gt_images[rgb_idx:depth_idx, :, :, :] = torch.from_numpy(gt_tar_depths).to(input_image).permute(0, 3, 1, 2)
            gt_images[sem_idx:in_depth_idx, :, :, :] = torch.from_numpy(gt_in_depths).to(input_image).permute(0, 3, 1, 2)
        else:
            torchvision.utils.save_image(torchvision.utils.make_grid(fake_tar_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/pred_tar_depths.png")
            torchvision.utils.save_image(torchvision.utils.make_grid(fake_in_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/pred_in_depths.png")
            torchvision.utils.save_image(torchvision.utils.make_grid(gt_tar_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/gt_tar_depths.png")
            torchvision.utils.save_image(torchvision.utils.make_grid(gt_in_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/gt_in_depths.png")

    elif ["rgb", "depth", "normal", "semantic"] == opt.prediction_types:
        num_target_rgbs, num_target_depths, num_input_depths = num_out_views, num_out_views, num_in_views
        rgb_idx = num_target_rgbs
        depth_idx = num_target_rgbs + num_target_depths
        normal_idx = depth_idx + num_out_views
        semantic_idx = normal_idx + num_out_views
        in_depth_idx = semantic_idx + num_input_depths
        in_normal_idx = in_depth_idx + num_in_views
        in_semantic_idx = in_normal_idx + num_in_views
        fake_tar_rgbs = torch.from_numpy(pred_images[:rgb_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        fake_tar_depths = torch.from_numpy(pred_images[rgb_idx:depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x1xhxw
        fake_tar_normals = torch.from_numpy(pred_images[depth_idx:normal_idx, :, :, :]).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        fake_tar_semantics = torch.from_numpy(pred_images[normal_idx:semantic_idx, :, :, :]).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        fake_in_depths = torch.from_numpy(pred_images[semantic_idx:in_depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x1xhxw
        fake_in_normals = torch.from_numpy(pred_images[in_depth_idx:in_normal_idx, :, :, :]).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        fake_in_semantics = torch.from_numpy(pred_images[in_normal_idx:in_semantic_idx, :, :, :]).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        gt_tar_rgbs = rearrange(batch["image_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,3,h,w
        gt_tar_depths = rearrange(batch["depth_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,1,h,w
        gt_tar_semantics = rearrange(batch["semantic_target"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # BxT,3,h,w
        gt_in_depths = rearrange(batch["depth_input"][0:1], "b t c h w -> (b t) c h w", t=num_in_views).cpu()[:, 0:1, :, :]   # B,1,h,w
        gt_in_semantics = rearrange(batch["semantic_input"][0:1], "b t c h w -> (b t) c h w", t=num_out_views).cpu()  # B,T_in,3,h,w
        pose_in = batch["pose_in"][0:1]  # B,T_in,4,4
        pose_out = batch["pose_out"][0:1]  # B,T_out,4,4
        min_depth = batch["depth_min"][0:1].cpu()  # B,1
        max_depth = batch["depth_max"][0:1].cpu()
        scene_scale = batch["scene_scale"][0:1].cpu()
        intrinsic_mat = batch["intrinsic"][0].cpu().numpy()
        print("validating RGB-D-S-N task on room {}".format(room_uid))

        save_input_output_pointcloud_with_sem(
            input_images=input_image.float(),
            input_depths=fake_in_depths.float(),
            input_semantics=fake_in_semantics.float(),
            output_images=fake_tar_rgbs.float(),
            output_depths=fake_tar_depths.float(),
            output_semantics=fake_tar_semantics.float(),
            poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=num_in_views),
            poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=num_out_views),
            min_depth=min_depth.float(),
            max_depth=max_depth.float(),
            scene_scale=scene_scale.float(),
            intrinsic_mat=intrinsic_mat,
            output_folder=output_folder,
            is_gt=False,
        )
        save_input_output_pointcloud_with_sem(
            input_images=input_image.float(),
            input_depths=gt_in_depths.float(),
            input_semantics=gt_in_semantics.float(),
            output_images=gt_tar_rgbs.float(),
            output_depths=gt_tar_depths.float(),
            output_semantics=gt_tar_semantics.float(),
            poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=num_in_views),
            poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=num_out_views),
            min_depth=min_depth.float(),
            max_depth=max_depth.float(),
            scene_scale=scene_scale.float(),
            intrinsic_mat=intrinsic_mat,
            output_folder=output_folder,
            is_gt=True,
        )

        # save prediceted depths of src and target views
        # convert depth into colorful depth
        fake_tar_depths = pred_images[rgb_idx:depth_idx, :, :, 0]
        fake_tar_depths = apply_depth_to_colormap(fake_tar_depths, cmap="viridis")  # BxT_out,H,W,3
        pred_tar_depths = torch.from_numpy(fake_tar_depths).permute(0, 3, 1, 2)
        pred_tar_depths = torchvision.utils.make_grid(pred_tar_depths, nrow=1)
        torchvision.utils.save_image(pred_tar_depths, f"{output_folder}/pred_tar_depths.png")
        pred_images[rgb_idx:depth_idx, :, :, :] = fake_tar_depths
        fake_in_depths = pred_images[semantic_idx:in_depth_idx, :, :, 0]
        fake_in_depths = apply_depth_to_colormap(fake_in_depths, cmap="viridis")  # BxT_in,H,W,3
        pre_in_depths = torch.from_numpy(fake_in_depths).permute(0, 3, 1, 2)
        pre_in_depths = torchvision.utils.make_grid(pre_in_depths, nrow=1)
        torchvision.utils.save_image(pre_in_depths, f"{output_folder}/pred_in_depths.png")
        pred_images[semantic_idx:in_depth_idx, :, :, :] = fake_in_depths
        # save predicted normals and semantics
        pred_tar_normals = torchvision.utils.make_grid(fake_tar_normals, nrow=1)
        torchvision.utils.save_image(pred_tar_normals, f"{output_folder}/pred_tar_normals.png")
        pred_in_normals = torchvision.utils.make_grid(fake_in_normals, nrow=1)
        torchvision.utils.save_image(pred_in_normals, f"{output_folder}/pred_in_normals.png")
        pred_tar_semantics = torchvision.utils.make_grid(fake_tar_semantics, nrow=1)
        torchvision.utils.save_image(pred_tar_semantics, f"{output_folder}/pred_tar_semantics.png")
        pred_in_semantics = torchvision.utils.make_grid(fake_in_semantics, nrow=1)
        torchvision.utils.save_image(pred_in_semantics, f"{output_folder}/pred_in_semantics.png")

        gt_tar_depths = ((gt_tar_depths + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].float().numpy()
        gt_tar_depths = apply_depth_to_colormap(gt_tar_depths, cmap="viridis")  # BxT_out,H,W,3
        torchvision.utils.save_image(
            torchvision.utils.make_grid(torch.from_numpy(gt_tar_depths).permute(0, 3, 1, 2), nrow=1),
            f"{output_folder}/gt_tar_depths.png",
        )
        gt_tar_depths = gt_tar_depths * 2 - 1.0
        gt_in_depths = ((gt_in_depths + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].float().numpy()
        gt_in_depths = apply_depth_to_colormap(gt_in_depths, cmap="viridis")  # BxT_in,H,W,3
        gt_images[rgb_idx:depth_idx, :, :, :] = torch.from_numpy(gt_tar_depths).to(input_image).permute(0, 3, 1, 2)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(torch.from_numpy(gt_in_depths).permute(0, 3, 1, 2), nrow=1),
            f"{output_folder}/gt_in_depths.png",
        )
        gt_in_depths = gt_in_depths * 2 - 1.0
        gt_images[semantic_idx:in_depth_idx, :, :, :] = torch.from_numpy(gt_in_depths).to(input_image).permute(0, 3, 1, 2)

    return pred_images, gt_images


@torch.no_grad()
def log_validation(
    validation_dataloader,
    vae,
    unet,
    args,
    accelerator,
    weight_dtype,
    opt: Options,
    split="val",
    LPIPS=None,
    ray_encoder=None,
    tokenizer=None,
    text_encoder=None,
    depth_vae=None,
    num_tasks=2,
    global_steps=0,
    exp_dir:str="",
):
    logger.info("Running {} validation... ".format(split))

    if not opt.edm_style_training:
        if opt.noise_scheduler_type == "ddim":
            noise_scheduler = DDIMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
        elif "dpmsolver" in opt.noise_scheduler_type:
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
            noise_scheduler.config.algorithm_type = opt.noise_scheduler_type
        else:
            raise NotImplementedError  # TODO: support more noise schedulers
    else:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule
        
    pipeline = SpatialGenDiffusionPipeline(
        vae=vae,
        depth_vae=depth_vae,
        ray_encoder=accelerator.unwrap_model(ray_encoder).eval() if ray_encoder is not None else None,
        unet=accelerator.unwrap_model(unet).eval(),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        feature_extractor=None,
        scheduler=noise_scheduler,
        safety_checker=None,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    val_lpips = 0
    val_ssim = 0
    val_psnr = 0
    val_loss = 0
    val_num = 0
    
    LPIPS = LPIPS.to('cpu')

    output_dir = os.path.join(exp_dir, f"{split}")
    os.makedirs(output_dir, exist_ok=True)
    for valid_step, batch in enumerate(validation_dataloader):
        if args.max_val_steps is not None and valid_step >= args.max_val_steps:
            break
        
        T_in = batch["image_input"].shape[1]
        T_in_val = T_in
        T_out = batch["image_target"].shape[1]
        input_image = batch["image_input"][0:1, :T_in].to(dtype=weight_dtype)  # B,T,3,H,W
        input_image = rearrange(input_image, "b t c h w -> (b t) c h w", t=T_in)
        room_uid = batch["room_uid"][0].replace("/", "_")
            
        if opt.use_layout_prior:
            input_layout_sem_images = batch["semantic_layout_input"][0:1].to(weight_dtype)   # B,T,3,H,W
            target_layout_sem_images = batch["semantic_layout_target"][0:1].to(weight_dtype)  # B,T,3,H,W
            input_layout_depth_images = batch["depth_layout_input"][0:1].to(weight_dtype)   # B,T,3,H,W
            target_layout_depth_images = batch["depth_layout_target"][0:1].to(weight_dtype)  # B,T,3,H,W
            input_layout_sem_images = rearrange(input_layout_sem_images, "b t c h w -> (b t) c h w", t=T_in)
            target_layout_sem_images = rearrange(target_layout_sem_images, "b t c h w -> (b t) c h w", t=T_out)
            input_layout_depth_images = rearrange(input_layout_depth_images, "b t c h w -> (b t) c h w", t=T_in)
            target_layout_depth_images = rearrange(target_layout_depth_images, "b t c h w -> (b t) c h w", t=T_out)
        else:
            input_layout_sem_images = None
            target_layout_sem_images = None
            input_layout_depth_images = None
            target_layout_depth_images = None
        # compose gt images
        if opt.prediction_types == ["rgb", "depth", "normal", "semantic"]:
            target_images = torch.cat(
                [
                    batch["image_target"][0:1],
                    batch["depth_target"][0:1],
                    batch["normal_target"][0:1],
                    batch["semantic_target"][0:1],
                ],
                dim=0,
            ).to(dtype=weight_dtype)  # 4B,T,3,H,W
        elif opt.prediction_types == ["rgb", "depth", "normal"]:
            target_images = torch.cat(
                [
                    batch["image_target"][0:1],
                    batch["depth_target"][0:1],
                    batch["normal_target"][0:1],
                ],
                dim=0,
            ).to(dtype=weight_dtype)  # 3B,T,3,H,W
        elif opt.prediction_types == ["rgb", "depth", "semantic"]:
            target_images = torch.cat(
                [
                    batch["image_target"][0:1],
                    batch["depth_target"][0:1],
                    batch["semantic_target"][0:1],
                ],
                dim=0,
            ).to(dtype=weight_dtype)  # 3B,T,3,H,W
        elif opt.prediction_types == ["rgb", "depth"]:
            target_images = torch.cat([batch["image_target"][0:1], batch["depth_target"][0:1]], dim=0).to(dtype=weight_dtype)  # 2B,T,3,H,W
        elif opt.prediction_types == ["rgb", "normal"]:
            target_images = torch.cat([batch["image_target"][0:1], batch["normal_target"][0:1]], dim=0).to(dtype=weight_dtype)  # 2B,T,3,H,W
        elif opt.prediction_types == ["rgb", "semantic"]:
            target_images = torch.cat([batch["image_target"][0:1], batch["semantic_target"][0:1]], dim=0).to(dtype=weight_dtype)  # 2B,T,3,H,W
        elif opt.prediction_types == ["rgb"]:
            target_images = batch["image_target"][0:1].to(dtype=weight_dtype)
            
        gt_image = rearrange(target_images, "b t c h w -> (b t) c h w", t=T_out)  # (num_tasks*BxT),3,H,W
        if "depth" in opt.prediction_types:
            input_depths = batch["depth_input"][0:1, :T_in].to(dtype=weight_dtype)  # B,T_in,3,H,W
            input_depths = rearrange(input_depths, "b t c h w -> (b t) c h w", t=T_in)

        if opt.prediction_types == ["rgb", "depth", "normal", "semantic"]:
            input_normals = rearrange(batch["normal_input"][0:1, :T_in], "b t c h w -> (b t) c h w", t=T_in).to(dtype=weight_dtype)  # (BxT_in),3,H,W
            input_semantics = rearrange(batch["semantic_input"][0:1, :T_in], "b t c h w -> (b t) c h w", t=T_in).to(dtype=weight_dtype)  # (BxT_in),3,H,W
            gt_image = torch.cat([gt_image, input_depths, input_normals, input_semantics], dim=0)  # (4BxT_out + 3BxT_in),3,H,W
        elif opt.prediction_types == ["rgb", "depth", "normal"]:
            input_normals = rearrange(batch["normal_input"][0:1, :T_in], "b t c h w -> (b t) c h w", t=T_in).to(dtype=weight_dtype)  # (BxT_in),3,H,W
            gt_image = torch.cat([gt_image, input_depths, input_normals], dim=0)  # (3BxT_out + 2BxT_in),3,H,W
        elif opt.prediction_types == ["rgb", "depth", "semantic"]:
            input_sems = rearrange(batch["semantic_input"][0:1, :T_in], "b t c h w -> (b t) c h w", t=T_in).to(dtype=weight_dtype)  # (BxT_in),3,H,W
            gt_image = torch.cat([gt_image, input_depths, input_sems], dim=0)  # (3BxT_out + 2BxT_in),3,H,W
        elif opt.prediction_types == ["rgb", "depth"]:
            gt_image = torch.cat([gt_image, input_depths], dim=0)  # (2BxT_out + T_in),3,H,W
        elif opt.prediction_types == ["rgb", "normal"]:
            input_normals = rearrange(batch["normal_input"][0:1, :T_in], "b t c h w -> (b t) c h w", t=T_in).to(dtype=weight_dtype)  # (BxT_in),3,H,W
            gt_image = torch.cat([gt_image, input_normals], dim=0)  # (2BxT_out + T_in),3,H,W
        elif opt.prediction_types == ["rgb", "semantic"]:
            input_sems = rearrange(batch["semantic_input"][0:1, :T_in], "b t c h w -> (b t) c h w", t=T_in).to(dtype=weight_dtype)  # (BxT_in),3,H,W
            gt_image = torch.cat([gt_image, input_sems], dim=0)  # (2BxT_out + T_in),3,H,W
        elif opt.prediction_types == ["rgb"]:
            gt_image = gt_image
            
        # compose task embeddings
        if opt.prediction_types == ["rgb", "depth", "normal", "semantic"]:
            task_embeddings = torch.cat(
                [
                    batch["color_task_embeddings"][0:1],
                    batch["depth_task_embeddings"][0:1],
                    batch["normal_task_embeddings"][0:1],
                    batch["semantic_task_embeddings"][0:1],
                ],
                dim=0,
            ).to(
                dtype=weight_dtype
            )  # 3B, T_in+T_out, 4
        elif opt.prediction_types == ["rgb", "depth", "normal"]:
            task_embeddings = torch.cat(
                [
                    batch["color_task_embeddings"][0:1],
                    batch["depth_task_embeddings"][0:1],
                    batch["normal_task_embeddings"][0:1],
                ],
                dim=0,
            ).to(
                dtype=weight_dtype
            )  # 3B, T_in+T_out, 4
        elif opt.prediction_types == ["rgb", "depth", "semantic"]:
            task_embeddings = torch.cat(
                [
                    batch["color_task_embeddings"][0:1],
                    batch["depth_task_embeddings"][0:1],
                    batch["semantic_task_embeddings"][0:1],
                ],
                dim=0,
            ).to(
                dtype=weight_dtype
            )  # 3B, T_in+T_out, 4
        elif opt.prediction_types == ["rgb", "depth"]:
            task_embeddings = torch.cat(
                [
                    batch["color_task_embeddings"][0:1],
                    batch["depth_task_embeddings"][0:1],
                ],
                dim=0,
            ).to(
                dtype=weight_dtype
            )  # 2B, T_in+T_out, 4
        elif opt.prediction_types == ["rgb", "normal"]:
            task_embeddings = torch.cat(
                [
                    batch["color_task_embeddings"][0:1],
                    batch["normal_task_embeddings"][0:1],
                ],
                dim=0,
            ).to(
                dtype=weight_dtype
            )  # 2B, T_in+T_out, 4
        elif opt.prediction_types == ["rgb", "semantic"]:
            task_embeddings = torch.cat(
                [
                    batch["color_task_embeddings"][0:1],
                    batch["semantic_task_embeddings"][0:1],
                ],
                dim=0,
            ).to(
                dtype=weight_dtype
            )  # 2B, T_in+T_out, 4
        elif opt.prediction_types == ["rgb"]:
            task_embeddings = batch["color_task_embeddings"][0:1].to(dtype=weight_dtype)  # B, T_in+T_out, 4
        if opt.use_layout_prior:
            task_embeddings = torch.cat(
                [task_embeddings, batch["layout_sem_task_embeddings"][0:1], batch["layout_depth_task_embeddings"][0:1]], dim=0)           
        
        task_embeddings = rearrange(task_embeddings, "b t c -> (b t) c", t=T_in + T_out).contiguous()  # 3B,(T_in+T_out), 4
        # logger.info(f"task_embeddings shape: {task_embeddings.shape}")

        # prepare warpped target images
        if opt.input_concat_warpped_image:
            # TODO: here we should first  generate the input_view depth map, then do the warpping. 
            # Temporarily use the GT input depth map!!!
            target_poses = batch["pose_out"][0:1].to(weight_dtype)  # B,T_out,4,4
            min_depth = batch["depth_min"][0:1].to(weight_dtype)  # B,1
            max_depth = batch["depth_max"][0:1].to(weight_dtype)
            scene_scale = batch["scene_scale"][0:1].to(weight_dtype)
            intrinsics = batch["intrinsic"][0:1].to(weight_dtype)
            # warp the depth maps to target views
            warpped_target_images: Float[Tensor, "B T 3 H W"] = cross_viewpoint_rendering_pt3d(input_rgbs=rearrange(input_image, "(b t) c h w -> b t c h w", t=T_in),
                                                                input_depths=rearrange(input_depths, "(b t) c h w -> b t c h w", t=T_in),
                                                                poses_output=target_poses,
                                                                intrinsic=intrinsics,
                                                                min_depth=min_depth,
                                                                max_depth=max_depth,
                                                                scene_scale=scene_scale,
                                                                use_scene_coord_map=opt.use_scene_coord_map,
                                                                )
            warpped_target_images = (warpped_target_images * 2. - 1.).clamp(-1., 1.)
            warpped_target_images = rearrange(warpped_target_images, "b t c h w -> (b t) c h w", t=T_out)  # BxT_out,3,H,W
        else:
            warpped_target_images = None
        
        # compose view indices
        # (B, T_in)  (B, T_out)
        input_rgb_indices, condition_indices, input_view_indices, target_view_indices, prediction_indices = compose_view_indices(batch, opt)
            
        # process rays and resize them to the same size as the latent image
        input_rays = torch.cat([batch["plucker_rays_input"][0:1, :T_in]] * num_tasks).to(dtype=weight_dtype)  # 2B,T_in,6,h,w
        out_rays = torch.cat([batch["plucker_rays_target"][0:1]] * num_tasks).to(dtype=weight_dtype)  # 2B,T_out,6,h,w

        images = []
        h, w = input_image.shape[2:]
        with torch.autocast("cuda", torch.float16):
            for guidance_scale in sorted(args.val_guidance_scales):
                # warpped_target_images = None
                image = pipeline(
                    input_imgs=input_image,
                    prompt_imgs=input_image,
                    input_indices=input_view_indices,  # (2B x T_in)
                    input_rgb_indices=input_rgb_indices,  # (B x T_in)
                    condition_indices=condition_indices,  # (B x T_in + 2B x (T_in + T_out))
                    target_indices=target_view_indices,  # (2B x T_out)
                    output_indices=prediction_indices,  # (2B x T_out + T_in)
                    input_rays=input_rays,
                    target_rays=out_rays,
                    task_embeddings=task_embeddings,
                    warpped_target_rgbs=warpped_target_images,
                    torch_dtype=weight_dtype,
                    height=h,
                    width=w,
                    T_in=T_in,
                    T_out=T_out,
                    guidance_scale=guidance_scale,
                    num_inference_steps=opt.num_inference_steps,
                    generator=generator,
                    output_type="numpy",
                    num_tasks=num_tasks,
                    cond_input_layout_sem_images=input_layout_sem_images,
                    cond_target_layout_sem_images=target_layout_sem_images,
                    cond_input_layout_depth_images=input_layout_depth_images,
                    cond_target_layout_depth_images=target_layout_depth_images,
                ).images

                output_folder = os.path.join(output_dir, room_uid + '_' + str(guidance_scale))
                os.makedirs(output_folder, exist_ok=True)
                
                # save warpped target rgbs
                if opt.input_concat_warpped_image:
                    warpped_target_rgbs = (warpped_target_images + 1.0) / 2.0
                    warpped_target_rgbs = torchvision.utils.make_grid(warpped_target_rgbs, nrow=1)
                    torchvision.utils.save_image(warpped_target_rgbs, f"{output_folder}/warpped_target_rgbs.png")
                # save layout condition images
                if opt.use_layout_prior:
                    in_lay_sem_images = (input_layout_sem_images + 1.0) / 2.0
                    in_lay_sem_images = torchvision.utils.make_grid(in_lay_sem_images, nrow=1)
                    torchvision.utils.save_image(in_lay_sem_images, f"{output_folder}/input_layout_semantics.png")
                    
                    tar_lay_sem_images = (target_layout_sem_images + 1.0) / 2.0
                    tar_lay_sem_images = torchvision.utils.make_grid(tar_lay_sem_images, nrow=1)
                    torchvision.utils.save_image(tar_lay_sem_images, f"{output_folder}/target_layout_semantics.png")
                    
                    if not opt.use_scene_coord_map:
                        in_lay_depth_images = ((input_layout_depth_images + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].cpu().numpy()
                        in_lay_depth_images = save_color_depth_image(in_lay_depth_images, output_path=f"{output_folder}/input_layout_depths.png") 
                        
                        tar_lay_depth_images = ((target_layout_depth_images + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].cpu().numpy()
                        tar_lay_depth_images = save_color_depth_image(tar_lay_depth_images, output_path=f"{output_folder}/target_layout_depths.png") 
                    else:
                        in_lay_depth_images = (input_layout_depth_images + 1.0) / 2.0
                        torchvision.utils.save_image(torchvision.utils.make_grid(in_lay_depth_images, nrow=1), f"{output_folder}/input_layout_depths.png")
                        tar_lay_depth_images = (target_layout_depth_images + 1.0) / 2.0
                        torchvision.utils.save_image(torchvision.utils.make_grid(tar_lay_depth_images, nrow=1), f"{output_folder}/target_layout_depths.png")
                    

                image, gt_image = logging_mv_mm_images(opt, batch=batch, pred_images=image, gt_images=gt_image, 
                                     input_image=input_image, num_in_views=T_in, num_out_views=T_out,
                                     output_folder=output_folder)
                # calc metrics
                pred_image = torch.from_numpy(image).permute(0, 3, 1, 2)
                images.append(pred_image)

                # only calculate image metric foor rgbs
                num_target_rgbs = T_out
                pred_rgbs = pred_image[0:num_target_rgbs]
                gt_rgbs = (gt_image[0:num_target_rgbs] / 2 + 0.5).clamp(0, 1).cpu()
                    
                        
                # pixel loss
                loss = tF.mse_loss(pred_rgbs, gt_rgbs).item()
                # LPIPS
                lpips = LPIPS(pred_rgbs, gt_rgbs)  # [-1, 1] torch tensor
                lpips = lpips.mean().item()
                # SSIM
                ssim = torch.tensor(calculate_ssim(
                    (rearrange(pred_rgbs, "bv c h w -> (bv c) h w").cpu().float().numpy() * 255.).astype(np.uint8),
                    (rearrange(gt_rgbs, "bv c h w -> (bv c) h w").cpu().float().numpy() * 255.).astype(np.uint8),
                    channel_axis=0,
                ), device=pred_rgbs.device)
                # PSNR
                psnr = -10. * torch.log10(tF.mse_loss(pred_rgbs, gt_rgbs))
                psnr = psnr.mean().item()

                val_loss += loss
                val_lpips += lpips
                val_ssim += ssim
                val_psnr += psnr

                val_num += 1

        image_logs.append(
            {
                "gt_image": gt_image / 2 + 0.5,
                "pred_images": images,
                "input_image": input_image / 2 + 0.5,
            }
        )

    pixel_loss = val_loss / val_num
    pixel_lpips = val_lpips / val_num
    pixel_ssim = val_ssim / val_num
    pixel_psnr = val_psnr / val_num

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log_id, log in enumerate(image_logs):
                pred_images = log["pred_images"]
                input_image = log["input_image"]
                gt_image = log["gt_image"]
                formatted_images = [[] for i in range(2 + len(pred_images))]
                # save to tensorboard
                input_image = torchvision.utils.make_grid(input_image, nrow=T_in_val)
                formatted_images[0].append(input_image)
                gt_image = torchvision.utils.make_grid(gt_image, nrow=T_out)
                formatted_images[1].append(gt_image)
                for sample_id, pred_image in enumerate(pred_images):
                    pred_image = torchvision.utils.make_grid(pred_image, nrow=T_out)
                    formatted_images[2 + sample_id].append(pred_image)

                # for sample_id, (input_image, gt_image, pred_image0, pred_image1 ) in enumerate(zip(*formatted_images)):
                for sample_id, (input_image, gt_image, pred_image0 ) in enumerate(zip(*formatted_images)):
                    input_image = input_image.float().detach().cpu()
                    tracker.writer.add_image(
                        f"{split}/{log_id}/input_{sample_id}",
                        input_image,
                        global_step=valid_step,
                    )
                    gt_image = gt_image.float().detach().cpu()
                    tracker.writer.add_image(
                        f"{split}/{log_id}/target_{sample_id}",
                        gt_image,
                        global_step=valid_step,
                    )
                    pred_image0 = pred_image0.float().detach().cpu()
                    tracker.writer.add_image(
                        f"{split}/{log_id}/pred_{sample_id}_0",
                        pred_image0,
                        global_step=valid_step,
                    )
                    # pred_image1 = pred_image1.float().detach().cpu()
                    # tracker.writer.add_image(
                    #     f"{split}/{log_id}/pred_{sample_id}_1",
                    #     pred_image1,
                    #     global_step=valid_step,
                    # )
                    # pred_image2 = pred_image2.float().detach().cpu()
                    # tracker.writer.add_image(
                    #     f"{split}/{log_id}/pred_{sample_id}_2",
                    #     pred_image2,
                    #     global_step=valid_step,
                    # )

            tracker.writer.add_scalar(f"{split}/T{T_in_val}_pixel_loss", pixel_loss, global_steps)
            tracker.writer.add_scalar(f"{split}/T{T_in_val}_lpips", pixel_lpips, global_steps)
            tracker.writer.add_scalar(f"{split}/T{T_in_val}_ssim", pixel_ssim, global_steps)
            tracker.writer.add_scalar(f"{split}/T{T_in_val}_psnr", pixel_psnr, global_steps)

    # after validation, set the pipeline back to training mode
    unet.train()
    vae.eval()
    if ray_encoder is not None:
        ray_encoder.train()
    LPIPS = LPIPS.to(accelerator.device)

    return image_logs

def encode_text(text_prompt, tokenizer, text_encoder, weight_dtype):
    text_inputs = tokenizer(
        text_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    text_embeds = text_encoder(text_input_ids)[0].to(weight_dtype)
    return text_embeds

def prediction_to_x0(noise_scheduler, 
                     noisy_latents,
                     model_prediction, 
                     timesteps, 
                     sigmas,
                     latent_scaling_factor, 
                     latent_shift_factor,
                     weight_dtype):
    if isinstance(noise_scheduler, DDPMScheduler) or isinstance(noise_scheduler, DDIMScheduler):
        alpha_prod_t = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps]  # (B*V_all,)
        beta_prod_t = 1. - alpha_prod_t  # (B*V_all,)
        while alpha_prod_t.ndim < model_prediction.ndim:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            beta_prod_t = beta_prod_t.unsqueeze(-1)
        if noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
            pred_x0_latents = model_prediction
        elif noise_scheduler.config.prediction_type == "epsilon":
            pred_x0_latents = (noisy_latents - beta_prod_t.sqrt() * model_prediction) / alpha_prod_t.sqrt()
        elif noise_scheduler.config.prediction_type == "v_prediction":
            pred_x0_latents = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_prediction
        else:
            raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
    elif isinstance(noise_scheduler, EulerDiscreteScheduler):
        if noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
            pred_x0_latents = model_prediction
        elif noise_scheduler.config.prediction_type == "epsilon":
            pred_x0_latents = model_prediction * (-sigmas) + noisy_latents
        elif noise_scheduler.config.prediction_type == "v_prediction":
            pred_x0_latents = model_prediction * (-sigmas / (sigmas**2 + 1) ** 0.5) + (noisy_latents / (sigmas**2 + 1))
        else:
            raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
    else:
        raise NotImplementedError  # TODO: support more noise schedulers

    # Render the predicted latents
    pred_x0_latents = pred_x0_latents / latent_scaling_factor + latent_shift_factor

    return pred_x0_latents.to(weight_dtype)
 
def vae_encode_image(vae: AutoencoderKL, images: torch.Tensor, num_views: int = 1, weight_dtype: torch.dtype = torch.float32):
    """Encode images to latents using the VAE encoder."""
    images = rearrange(images.to(weight_dtype), "b t c h w -> (b t) c h w", t=num_views)
    latents = vae.encode(images).latent_dist.mode() * vae.config.scaling_factor  # BT,4,H//8,W//8
    return latents

def vae_decode_image(vae: Union[AutoencoderKL, AutoencoderTiny], latents: torch.Tensor, num_views: int = 1, weight_dtype: torch.dtype = torch.float32, use_tiny_vae: bool = False, return_conf_map: bool = False):
    """Decode latents to images using the VAE decoder.
    Args:
        latents (torch.Tensor): Latents to decode.
        num_views (int): Number of views.
        weight_dtype (torch.dtype): Weight dtype.
        use_tiny_vae (bool): Whether to use the tiny VAE decoder.
    Returns:
        torch.Tensor: Decoded images, [-1, 1]
    """
    latents = latents / vae.config.scaling_factor if not use_tiny_vae else latents
    images = vae.decode(latents).sample
    if return_conf_map:
        conf_map = images[:, 3:4, :, :]  # BxT,1,H,W
        images = images[:, :3, :, :]
        images = rearrange(images, "(b t) c h w -> b t c h w", t=num_views)
        conf_map = rearrange(conf_map, "(b t) c h w -> b t c h w", t=num_views)
        return images.to(weight_dtype), conf_map.to(weight_dtype)
    else:
        return rearrange(images, "(b t) c h w -> b t c h w", t=num_views).to(weight_dtype)  # B,T,3,H,W
        
def parse_in_view_depth_indice(prediction_types: List[str], batch: Dict[str, Tensor]) -> Tuple[int, int]:
    bsz = batch["image_target"].shape[0]
    T_in = batch["image_input"].shape[1]
    T_out = batch["image_target"].shape[1]
    # parse the depth map indices
    if prediction_types == ["rgb", "depth", "normal", "semantic"]:
        raise ValueError(f"{prediction_types} is not supported")
    elif prediction_types == ["rgb", "depth", "normal"]:
        raise ValueError(f"{prediction_types} is not supported")
    elif prediction_types == ["rgb", "depth", "semantic"]:
        target_view_indice = 0
        input_view_depth_indice = bsz * 3 * T_out
        input_view_semantic_indice = bsz * (3 * T_out + T_in)
    elif prediction_types == ["rgb", "depth"]:
        raise ValueError(f"{prediction_types} is not supported")
    elif prediction_types == ["rgb", "normal"]:
        raise ValueError(f"{prediction_types} is not supported")
    elif prediction_types == ["rgb", "semantic"]:
        raise ValueError(f"{prediction_types} is not supported")
    elif prediction_types == ["rgb"]:
        raise ValueError(f"{prediction_types} is not supported")
    else:
        raise ValueError(f"{prediction_types} is not supported")

    return input_view_depth_indice, input_view_semantic_indice      
      
def main():
    PROJECT_NAME = "SpatialGen"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D indoor scene generation",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",  # log_image currently only for wandb
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=-1,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=1,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[2.],
        help="CFG scale used for validation"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )
    parser.add_argument(
        "--load_pretrained_gsvae_model",
        type=str,
        default='/project/lrmcongen/codes/fangchuan/DiffSplat/out/gsvae_sd_spiral',
        help="Tag of a pretrained GSVAE in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsvae_model_ckpt",
        type=int,
        default=100000,
        help="Iteration of the pretrained GSVAE checkpoint"
    )
    parser.add_argument(
        "--mvd_warmup_steps",
        type=int,
        default=10000,
        help="Number of steps for the warmup in the MultiViewDiffusion model.",
    )
    parser.add_argument(
        "--use_tiny_vae",
        action="store_true",
        help="Whether to use tiny VAE for training.",
    )
    parser.add_argument(
        "--use_scm_conf_map",
        action="store_true",
        help="Whether to use confidence map of scene coord map for training.",
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)

    # Create an experiment directory using the `tag`
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    pipeline_dir = os.path.join(exp_dir, "pipeline")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pipeline_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # if use_layout_prior is True, we need to add the layout_sem and layout_depth channel to model tha multi-modality task
    num_tasks = len(opt.prediction_types) + 2 if opt.use_layout_prior else len(opt.prediction_types)
    
    # Load the training and validation dataset
    train_dataset = MixDataset(
        dataset_names=opt.dataset_names,
        spatialgen_data_dir=opt.spatialgen_data_dir,
        hypersim_data_dir=opt.hypersim_data_dir,
        structured3d_data_dir=None,
        split_filepath=opt.train_split_file,
        image_height=opt.input_res,
        image_width=opt.input_res,
        T_in=opt.num_input_views,
        total_view=opt.num_views,
        validation=False,
        sampler_type=opt.trajectory_sampler_type,
        use_normal='normal' in opt.prediction_types,
        use_semantic='semantic' in opt.prediction_types or opt.use_layout_prior,
        use_metric_depth=opt.use_metric_depth,
        use_scene_coord_map=opt.use_scene_coord_map,
        use_layout_prior=opt.use_layout_prior,
        return_metric_data=False,
    )
    val_dataset = MixDataset(
        dataset_names=opt.dataset_names,
        spatialgen_data_dir=opt.spatialgen_data_dir,
        hypersim_data_dir=opt.hypersim_data_dir,
        structured3d_data_dir=None,
        split_filepath=opt.train_split_file,
        image_height=opt.input_res,
        image_width=opt.input_res,
        T_in=opt.num_input_views,
        total_view=opt.num_views,
        validation=True,
        sampler_type=opt.trajectory_sampler_type,
        use_normal='normal' in opt.prediction_types,
        use_semantic='semantic' in opt.prediction_types or opt.use_layout_prior,
        use_metric_depth=opt.use_metric_depth,
        use_scene_coord_map=opt.use_scene_coord_map,
        use_layout_prior=opt.use_layout_prior,
        return_metric_data=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    logger.info(f"Load [{len(train_loader)}] training samples and [{len(val_loader)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # LPIPS loss
    if accelerator.is_main_process:
        _ = LPIPS(net="vgg")
        del _
    accelerator.wait_for_everyone()  # wait for pretrained backbone weights to be downloaded
    lpips_loss_fn = LPIPS(net="vgg").to(accelerator.device)
    lpips_loss_fn = lpips_loss_fn.requires_grad_(False)
    lpips_loss_fn.eval()

    # Initialize the model, optimizer and lr scheduler
    in_channels = 4  # hard-coded for SD 1.5/2.1
    if opt.input_concat_plucker:
        in_channels += 16
    if opt.input_concat_binary_mask:
        in_channels += 1
    if opt.input_concat_warpped_image:
        in_channels += 4
    unet_from_pretrained_kwargs = {
        "sample_size": opt.input_res // 8,  # `8` hard-coded for SD 1.5/2.1
        "in_channels": in_channels,
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
        "input_concat_warpped_image": opt.input_concat_warpped_image,
        "num_input_views": opt.num_input_views,
        "num_output_views": opt.num_views - opt.num_input_views,
        "num_tasks": num_tasks,
        "cd_attention_mid": num_tasks > 1 and opt.enable_mm_attn,
        "multiview_attention": opt.enable_mv_attn,
        "sparse_mv_attention": False,
        "disable_mv_attention_in_64x64": opt.input_res == 512,
    }
    unet, loading_info = UNetMVMM2DConditionModel.from_pretrained_new(opt.pretrained_model_name_or_path, subfolder="unet",
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True, **unet_from_pretrained_kwargs)
    logger.info(f"Loading info: {loading_info}\n")
    logger.info(f"opt.input_concat_plucker: {opt.input_concat_plucker}, opt.input_concat_binary_mask: {opt.input_concat_binary_mask}, opt.input_concat_warpped_image: {opt.input_concat_warpped_image}, opt.enable_mv_attn: {opt.enable_mv_attn}, opt.enable_mm_attn: {opt.enable_mm_attn}\n")
    
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
    if args.use_tiny_vae:
        scm_vae = AutoencoderTiny.from_pretrained('/data-nas/experiments/zhenqing/diffsplat/tinyvae-ckpt-wconf-016000')
        logger.info(f"Load SCM_VAE from /data-nas/experiments/zhenqing/diffsplat/tinyvae-ckpt-wconf-016000")
        assert args.use_scm_conf_map, "use_scm_conf_map should be True when using tiny vae"
    else:
        scm_vae = AutoencoderKL.from_pretrained('/data-nas/experiments/zhenqing/diffsplat/sdvae-wconf_ckpt-010000', subfolder=None)
        logger.info(f"Load SCM_VAE from /data-nas/experiments/zhenqing/diffsplat/sdvae-wconf_ckpt-010000")
        
    tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder")
    if os.path.exists(os.path.join(opt.pretrained_model_name_or_path, "ray_encoder")):
        ray_encoder = RayMapEncoder.from_pretrained_new(
            opt.pretrained_model_name_or_path,
            subfolder="ray_encoder",
        )
    else:
        ray_encoder_cfg = RayMapEncoderConfig(
            image_size=opt.input_res,
            patch_size=8,
        )
        ray_encoder = RayMapEncoder(config=ray_encoder_cfg)
        
    if not opt.edm_style_training:
        noise_scheduler = DDPMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        logger.info("Performing EDM-style training")
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    # adaptor
    cat3d_adpator = CAT3DAdaptor(        
                                 unet_type="UNetMVMM2DConditionModel",
                                 unet_config=unet.config,
                                 ray_encoder_config=ray_encoder.config,
                                 num_sample_views=opt.num_views, 
                                 prediction_types=opt.prediction_types, 
                                 use_layout_prior=opt.use_layout_prior,
                                 unet_in_channels=in_channels, 
                                 unet=unet, 
                                 ray_encoder=ray_encoder)

    if args.use_ema:
        ema_cat3d = MyEMAModel(
            cat3d_adpator.parameters(),
            model_cls=CAT3DAdaptor,
            # model_config=unet.config,
            **configs["train"]["ema_kwargs"]
        )

    # Freeze VAE and GSVAE
    vae.requires_grad_(False)
    vae.eval()
    scm_vae.requires_grad_(False)
    scm_vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    trainable_module_names = []
    if opt.trainable_modules is None:
        cat3d_adpator.requires_grad_(True)
    else:
        cat3d_adpator.requires_grad_(False)
        for name, module in cat3d_adpator.named_modules():
            for module_name in tuple(opt.trainable_modules.split(",")):
                if module_name in name:
                    for params in module.parameters():
                        params.requires_grad = True
                    trainable_module_names.append(name)
    logger.info(f"Trainable parameter names: {trainable_module_names}\n")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_cat3d.save_pretrained(os.path.join(output_dir, "ema_cat3d"))

        accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

    # enable efficient training
    enable_flash_attn_if_avail(unet)
    if opt.grad_checkpoint:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()
        scm_vae.enable_gradient_checkpointing()
    vae.enable_slicing()
    scm_vae.enable_slicing()
    
    unet_params, params_class_embedding, rayencoder_params = [], [], []
    for name, param in cat3d_adpator.named_parameters():
        if "unet.class_embedding" in name:
            params_class_embedding.append(param)
        elif "unet" in name:
            unet_params.append(param)
        elif "ray_encoder" in name:
            rayencoder_params.append(param)
    optimizer = get_optimizer(
        params=[
            {"params": unet_params, "lr": configs["optimizer"]["lr"]},
            {"params": params_class_embedding, "lr": configs["optimizer"]["lr"] * opt.lr_mult},
            {"params": rayencoder_params, "lr": configs["optimizer"]["lr"] * opt.lr_mult},
        ],
        **configs["optimizer"]
    )

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting    
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
        
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu


    # Prepare everything with `accelerator`
    cat3d_adpator, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        cat3d_adpator, optimizer, lr_scheduler, train_loader, val_loader
    )
    # Set classes explicitly for everything
    cat3d_adpator: DistributedDataParallel
    optimizer: AcceleratedOptimizer
    lr_scheduler: AcceleratedScheduler
    train_loader: DataLoaderShard
    val_loader: DataLoaderShard

    if args.use_ema:
        ema_cat3d.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move `text_encoder`, `vae` to gpu and cast to `weight_dtype`
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # gsvae.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    scm_vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config["resolution"] = 256
        tracker_config["prediction_types"] = "rgb"
        tracker_config["val_guidance_scales"] = 3.0
        accelerator.init_trackers(args.tag, config=tracker_config)
        
    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    num_train_epochs = configs["train"]["epochs"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    assert num_train_epochs * updated_steps_per_epoch == total_updated_steps
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{num_train_epochs}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")

    # (Optional) Load checkpoint
    global_update_step = 0
    first_epoch = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            # Get the most recent checkpoint
            dirs = [d for d in os.listdir(ckpt_dir) if len(os.listdir(osp.join(ckpt_dir, d))) > 0]
            dirs = sorted(dirs, key=lambda x: int(x))
            args.resume_from_iter = int(dirs[-1]) if len(dirs) > 0 else None
        if args.resume_from_iter is not None:
            logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
            # Load everything
            accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
            if args.use_ema:
                ema_cat3d.load_state_dict(torch.load(
                    os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}", "ema_cat3d.pth"),
                    map_location=accelerator.device,
                ))
            global_update_step = int(args.resume_from_iter)
            first_epoch = global_update_step // updated_steps_per_epoch

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)
        util.save_model_architecture(accelerator.unwrap_model(cat3d_adpator), exp_dir)


    def get_sigmas(timesteps: Tensor, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(dtype=dtype, device=accelerator.device)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Encode text embedding for prompt
    text_prompt = "indoor scene"
    precomp_text_embeds = encode_text(text_prompt, tokenizer, text_encoder, weight_dtype)
    null_text_prompt = ""
    null_text_embeds = encode_text(null_text_prompt, tokenizer, text_encoder, weight_dtype)
    # logger.info(f"text_embeds: {precomp_text_embeds.shape}, neg_text_embeds: {neg_text_embeds.shape}\n")
    
    # Start training
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        disable=not accelerator.is_main_process
    )
    for epoch in range(first_epoch, num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        # update the num_input_views and num_target_views
        if global_update_step >= args.mvd_warmup_steps:
            random_input_prob = torch.rand(1).item()
            if random_input_prob < 0.3:
                train_loader.dataset.set_num_input_views(1)
            elif random_input_prob >= 0.3 and random_input_prob <= 0.6:
                train_loader.dataset.set_num_input_views(3)
            # else:
            #     train_loader.dataset.set_num_input_views(7)
                                
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(cat3d_adpator):
                
                cat3d_adpator: CAT3DAdaptor
                
                bsz = batch["image_target"].shape[0]

                T_in = batch["image_input"].shape[1]
                T_out = batch["image_target"].shape[1]
                input_poses = batch["pose_in"].to(dtype=weight_dtype)  # B,T,4,4
                target_poses = batch["pose_out"].to(dtype=weight_dtype)  # B,T,4,4
                num_sample_views = T_in + T_out

                # intrinsics
                intrinsics = batch["intrinsic"].to(dtype=weight_dtype)  # (B, 3, 3)
                
                min_depth = batch["depth_min"].to(dtype=weight_dtype)  # B,1
                max_depth = batch["depth_max"].to(dtype=weight_dtype)  # B,1
                scene_scale = batch["scene_scale"].to(dtype=weight_dtype)  # B,1
                
                input_image = batch["image_input"].to(dtype=weight_dtype)  # B,T,3,H,W
                input_image = rearrange(input_image, "b t c h w -> (b t) c h w", t=T_in)
                input_latents: Float[Tensor, "BNt C H W"] = vae.encode(input_image).latent_dist.mode() * vae.config.scaling_factor  # BT,4,H//8,W//8
                # input_rgb_latents = rearrange(input_latents/vae.config.scaling_factor, "(b v) c h w -> b v c h w", v=T_in)

                # B*num_tasks*T_out + B*(num_tasks-1)*T_in, 4, H//8, W//8
                prediction_latents: Float[Tensor, "BNt C H W"] = cat3d_adpator.module.compose_prediction_latents(batch, weight_dtype, vae)
                # prediction_latents: Float[Tensor, "BNt C H W"] = cat3d_adpator.compose_prediction_latents(batch, weight_dtype, vae)
                
                if opt.input_concat_warpped_image:
                    # parse encoded depth image
                    in_view_depth_start_idx, in_view_depth_end_idx = parse_in_view_depth_indice(opt.prediction_types, batch)
                    encoded_depth_latents = prediction_latents[in_view_depth_start_idx:in_view_depth_end_idx, :, :, :]  # B*T_in,4,H//8,W//8
                    decoded_depth_maps, decoded_depth_confs = vae_decode_image(scm_vae, encoded_depth_latents, num_views=T_in, weight_dtype=weight_dtype, use_tiny_vae=args.use_tiny_vae, return_conf_map=args.use_scm_conf_map)  # B, T_in,3,H,W
                    # from torchvision.utils import save_image
                    # save_image((decoded_depth_maps*0.5+0.5)[:,0,...], "decoded_depth_maps.png", nrow=T_in)
                    # save_image(decoded_depth_confs[:,0,...], "decoded_depth_confs.png", nrow=T_in, normalize=True)
                    # logger.info(f"decoded_depth_confs min: {decoded_depth_confs.min()}, median: {decoded_depth_confs.median()}, max: {decoded_depth_confs.max()}\n")
                    depth_masks = (decoded_depth_confs > 5.0).to(decoded_depth_maps)
                    decoded_depth_maps = decoded_depth_maps * depth_masks
                    # warp the depth maps to target views
                    warpped_target_images: Float[Tensor, "B T 3 H W"] = cross_viewpoint_rendering_pt3d(input_rgbs=rearrange(input_image, "(b t) c h w -> b t c h w", t=T_in),
                                                                      input_depths=decoded_depth_maps,
                                                                      poses_output=target_poses,
                                                                      intrinsic=intrinsics,
                                                                      min_depth=min_depth,
                                                                      max_depth=max_depth,
                                                                      scene_scale=scene_scale,
                                                                      use_scene_coord_map=opt.use_scene_coord_map,
                                                                      )
                    # save_image(rearrange(warpped_target_images, "b t c h w -> (b t) c h w"), "warpped_target_images.png", nrow=1)
                    # save_image(input_image*0.5+0.5, "input_rgb.png", nrow=1)
                    # save_image(rearrange(batch["image_target"]*0.5+0.5, "b t c h w -> (b t) c h w"), "target_rgb.png", nrow=1)
                    warpped_target_images = (warpped_target_images * 2. - 1.).clamp(-1., 1.) 
                    warpped_img_latents: Float[Tensor, "BNt C H W"] = vae_encode_image(vae, warpped_target_images, num_views=T_out, weight_dtype=weight_dtype)
                
                if opt.use_layout_prior:
                    input_layout_sem_images = batch["semantic_layout_input"]   # B,T,3,H,W
                    target_layout_sem_images = batch["semantic_layout_target"]  # B,T,3,H,W
                    input_layout_sem_latents: Float[Tensor, "BNt C H W"] = vae_encode_image(vae, input_layout_sem_images, num_views=T_in, weight_dtype=weight_dtype)
                    target_layout_sem_latents: Float[Tensor, "BNt C H W"] = vae_encode_image(vae, target_layout_sem_images, num_views=T_out, weight_dtype=weight_dtype)
                    input_layout_depth_images = batch["depth_layout_input"]   # B,T,3,H,W
                    target_layout_depth_images = batch["depth_layout_target"]  # B,T,3,H,W
                    input_layout_depth_latents: Float[Tensor, "BNt C H W"] = vae_encode_image(vae, input_layout_depth_images, num_views=T_in, weight_dtype=weight_dtype)
                    target_layout_depth_latents: Float[Tensor, "BNt C H W"] = vae_encode_image(vae, target_layout_depth_images, num_views=T_out, weight_dtype=weight_dtype)
                    
                    layout_sem_latents = torch.cat([input_layout_sem_latents, target_layout_sem_latents], dim=0)  # B*(T_in+T_out), 4, H//8, W//8
                    layout_depth_latents = torch.cat([input_layout_depth_latents, target_layout_depth_latents], dim=0)  # B*(T_in+T_out), 4, H//8, W//8
                    layout_latents = torch.cat([layout_sem_latents, layout_depth_latents], dim=0)  # B*2*(T_in+T_out), 4, H//8, W//8


                task_embeddings = cat3d_adpator.module.compose_task_embeddings(batch, weight_dtype)
                # task_embeddings = cat3d_adpator.compose_task_embeddings(batch, weight_dtype)
                task_embeddings = torch.cat([torch.sin(task_embeddings), torch.cos(task_embeddings)], dim=-1)  # B*num_tasks*num_sample_views, 4                
                
                input_rgb_indices, condition_indices, input_indices, target_indices, output_indices = cat3d_adpator.module.compose_view_indices(batch)
                # input_rgb_indices, condition_indices, input_indices, target_indices, output_indices = cat3d_adpator.compose_view_indices(batch)
                
                # Sample noise that we'll add to the latents
                target_noise = torch.randn_like(prediction_latents)

                if not opt.edm_style_training:
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=prediction_latents.device).long()
                else:
                    # In EDM formulation, the model is conditioned on the pre-conditioned noise levels
                    # instead of discrete timesteps, so here we sample indices to get the noise levels
                    # from `scheduler.timesteps`
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=prediction_latents.device)
                timesteps_all = timesteps.repeat_interleave(num_sample_views*num_tasks)
                # make input conditions' timestep to 0
                timesteps_all[condition_indices] = 0
                timesteps_target = timesteps_all[output_indices]

                noisy_latents = noise_scheduler.add_noise(prediction_latents, target_noise, timesteps_target)
                if not opt.edm_style_training:
                    noisy_latents = noisy_latents
                else:
                    # For EDM-style training, we first obtain the sigmas based on the continuous timesteps
                    # Then precondition the final model inputs based on these sigmas instead of the timesteps
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364
                    sigmas = get_sigmas(timesteps_target, len(noisy_latents.shape), weight_dtype)
                    noisy_latents = noisy_latents / ((sigmas**2 + 1)**0.5)
                    

                # process rays and resize them to the same size as the latent image
                in_view_rays = batch["plucker_rays_input"].to(dtype=weight_dtype)  # B,T_in,6,h,w
                out_view_rays = batch["plucker_rays_target"].to(dtype=weight_dtype)  # B,T_out,6,h,w

                prom_text_embeds = precomp_text_embeds.repeat(bsz, 1, 1)
                neg_text_embeds = null_text_embeds.repeat(bsz, 1, 1)

                if opt.cfg_dropout_prob > 0.:
                    # Drop a group of multi-view images as a whole
                    random_p = torch.rand(bsz, device=prediction_latents.device)

                    # Sample masks for the conditioning VAE images
                    if opt.view_concat_condition:
                        image_mask_dtype = weight_dtype
                        image_mask = 1 - (
                            (random_p >= opt.cfg_dropout_prob).to(image_mask_dtype)
                            * (random_p < 3 * opt.cfg_dropout_prob).to(image_mask_dtype)
                        )  # actual dropout rate is 2 * `cfg.condition_drop_rate`
                        image_mask = image_mask.reshape(bsz, 1, 1, 1, 1)
                        # Final VAE image conditioning
                        input_latents = rearrange(input_latents, "(b t) c h w -> b t c h w", t=T_in)
                        input_latents = image_mask * input_latents
                        input_latents = rearrange(input_latents, "b t c h w -> (b t) c h w", t=T_in)
                        
                        if opt.input_concat_warpped_image:
                            # drop warpped latents
                            warp_latent_drop_mask = (1.0 - (random_p < 2 * opt.cfg_dropout_prob).to(image_mask_dtype))
                            warp_latent_drop_mask = warp_latent_drop_mask.reshape(bsz, 1, 1, 1, 1)
                            warpped_img_latents = rearrange(warpped_img_latents, "(b t) c h w -> b t c h w", t=T_out)
                            warpped_img_latents = warp_latent_drop_mask * warpped_img_latents
                            warpped_img_latents = rearrange(warpped_img_latents, "b t c h w -> (b t) c h w", t=T_out)
                        # 1 for input views, 0 for target views and masked inputs
                        images_in_masks = image_mask.repeat(1, T_in, 1, input_latents.shape[-2], input_latents.shape[-1]).to(dtype=weight_dtype)
                        images_in_masks = rearrange(images_in_masks, "b t c h w -> (b t) c h w", t=T_in)

                    # drop rays_embeds
                    ray_drop_mask = torch.Tensor([1.0]).to(input_latents)

                    # Sample masks for the conditioning text prompts
                    text_mask_dtype = weight_dtype
                    text_mask = 1 - (
                            (random_p >= opt.cfg_dropout_prob).to(text_mask_dtype)
                            * (random_p < 3 * opt.cfg_dropout_prob).to(text_mask_dtype)
                        )
                    text_mask = text_mask.reshape(bsz, 1, 1)
                    # Final text conditioning
                    bs_text_embeds = text_mask * prom_text_embeds + (1 - text_mask) * neg_text_embeds
                else:
                    images_in_masks = torch.ones((bsz *T_in, 1, input_latents.shape[-2], input_latents.shape[-1]), device=accelerator.device, dtype=weight_dtype)
                    ray_drop_mask = torch.Tensor([1.0]).to(input_latents)

                bs_text_embeds = bs_text_embeds.repeat(num_tasks * num_sample_views, 1, 1)

                # (B*num_tasks*T_out + B*(num_tasks-1)*T_in), 4, H//8, W//8
                condition_latents = torch.cat([input_latents, layout_latents], dim=0) if opt.use_layout_prior else input_latents
                prediction_on_target, latent_model_input = cat3d_adpator(
                    input_rgb_latents=input_latents,
                    condition_latents=condition_latents,
                    noisy_latents=noisy_latents,
                    warpped_target_rgb_latents=warpped_img_latents if opt.input_concat_warpped_image else None,
                    input_view_rays=in_view_rays,
                    target_view_rays=out_view_rays,
                    bs_text_embeds=bs_text_embeds,
                    timesteps_all=timesteps_all,
                    task_embeddings=task_embeddings,
                    input_rgb_indices=input_rgb_indices,
                    condition_indices=condition_indices,
                    input_view_indices=input_indices,
                    target_view_indices=target_indices,
                    prediction_indices=output_indices,
                    ray_drop_masks=ray_drop_mask,
                    cond_image_masks=images_in_masks,
                )
                
                if not opt.edm_style_training:
                    weighting = 1.
                    model_pred = prediction_on_target
                else:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364
                    if noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
                        model_pred = prediction_on_target
                    elif noise_scheduler.config.prediction_type == "epsilon":
                        model_pred = prediction_on_target * (-sigmas) + noisy_latents
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        model_pred = prediction_on_target * (-sigmas / (sigmas**2 + 1) ** 0.5) + (noisy_latents / (sigmas**2 + 1))
                    else:
                        raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
                    weighting = (sigmas**-2.).float()

                # Get the target for loss depending on the prediction type
                if opt.edm_style_training or noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
                    target = prediction_latents
                elif noise_scheduler.config.prediction_type == "epsilon":
                    target = target_noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(prediction_latents, target_noise, timesteps_target)
                else:
                    raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")

                if opt.snr_gamma <= 0.:
                    diffusion_loss = weighting * tF.mse_loss(model_pred.float(), target.float(), reduction="none")
                    diffusion_loss = diffusion_loss.mean(dim=list(range(1, len(diffusion_loss.shape))))
                else:
                    assert not opt.edm_style_training, "Min-SNR formulation is not supported when conducting EDM-style training"
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise/v instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps_target)
                    mse_loss_weights = torch.stack([snr, opt.snr_gamma * torch.ones_like(timesteps_target)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights /= snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights /= (1. + snr)
                    else:
                        raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    diffusion_loss = tF.mse_loss(model_pred.float(), target.float(), reduction="none")
                    diffusion_loss = mse_loss_weights * diffusion_loss.mean(dim=list(range(1, len(diffusion_loss.shape))))

                # Backpropagate
                accelerator.backward(diffusion_loss.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(cat3d_adpator.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_cat3d.step(cat3d_adpator.parameters())

                progress_bar.update(1)
                global_update_step += 1

                # Save checkpoint
                if (global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                    or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                    or global_update_step == total_updated_steps):  # 3. last step of an epoch

                    gc.collect()
                    if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                        accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                    elif accelerator.is_main_process:
                        accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                    accelerator.wait_for_everyone()  # ensure all processes have finished saving
                    if accelerator.is_main_process:
                        # save pipeline
                        if configs["train"]["ckpts_total_limit"] is not None:
                            pipelines = os.listdir(pipeline_dir)
                            pipelines = [d for d in pipelines if d.startswith("pipeline")]
                            pipelines = sorted(pipelines, key=lambda x: int(x.split("-")[1]))

                            # before we save the new pipeline, we need to have at _most_ `checkpoints_total_limit - 1` pipeline
                            if len(pipelines) >= configs["train"]["ckpts_total_limit"]:
                                num_to_remove = len(pipelines) - configs["train"]["ckpts_total_limit"] + 1
                                removing_pipelines = pipelines[0:num_to_remove]

                                logger.info(f"{len(pipelines)} pipelines already exist, removing {len(removing_pipelines)} pipelines")
                                logger.info(f"removing pipelines: {', '.join(removing_pipelines)}")

                                for removing_pipeline in removing_pipelines:
                                    removing_pipeline = os.path.join(pipeline_dir, removing_pipeline)
                                    shutil.rmtree(removing_pipeline)

                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_cat3d.store(cat3d_adpator.parameters())
                            ema_cat3d.copy_to(cat3d_adpator.parameters())
                        unwrapped_cat3d_adpator = accelerator.unwrap_model(cat3d_adpator)
                        pipeline = SpatialGenDiffusionPipeline(
                            vae=vae,
                            depth_vae=scm_vae,
                            unet=unwrapped_cat3d_adpator.unet,
                            ray_encoder=unwrapped_cat3d_adpator.ray_encoder,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            scheduler=noise_scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )
                        pipeline_save_path = os.path.join(pipeline_dir, f"pipeline-{global_update_step:06d}")
                        pipeline.save_pretrained(pipeline_save_path)

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_cat3d.restore(cat3d_adpator.parameters())

                    gc.collect()

                # Evaluate on the validation set
                if accelerator.is_main_process:
                    if (global_update_step == 1
                        or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                        or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                        or global_update_step == total_updated_steps):  # 4. last step of an epoch

                        torch.cuda.empty_cache()
                        gc.collect()

                        # Use EMA parameters for evaluation
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference
                            ema_cat3d.store(cat3d_adpator.parameters())
                            ema_cat3d.copy_to(cat3d_adpator.parameters())
                        # Perform validation on the validation set
                        image_logs = log_validation(
                            val_loader,
                            vae=vae,
                            unet=cat3d_adpator.module.unet,
                            # unet=cat3d_adpator.unet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            opt=opt,
                            split="val",
                            LPIPS=lpips_loss_fn,
                            ray_encoder=cat3d_adpator.module.ray_encoder,
                            # ray_encoder=cat3d_adpator.ray_encoder,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            depth_vae=scm_vae,
                            num_tasks=num_tasks,
                            global_steps=global_update_step,
                            exp_dir=exp_dir,
                        )
                        
                        # Perform validation on the training set
                        image_logs = log_validation(
                            train_loader,
                            vae=vae,
                            unet=cat3d_adpator.module.unet,
                            # unet=cat3d_adpator.unet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            opt=opt,
                            split="train",
                            LPIPS=lpips_loss_fn,
                            ray_encoder=cat3d_adpator.module.ray_encoder,
                            # ray_encoder=cat3d_adpator.ray_encoder,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            depth_vae=scm_vae,
                            num_tasks=num_tasks,
                            global_steps=global_update_step,
                            exp_dir=exp_dir,
                        )

                        if args.use_ema:
                            # Switch back to the original UNet parameters
                            ema_cat3d.restore(cat3d_adpator.parameters())

                        torch.cuda.empty_cache()
                        gc.collect()

            # Log the training progress
            if global_update_step % configs["train"]["log_freq"] == 0 or global_update_step == 1 \
                or global_update_step % updated_steps_per_epoch == 0:  # last step of an epoch
                log_loss = diffusion_loss.mean().detach().item()
                loss_epoch += log_loss
                num_train_elems += 1
                logs = {
                    "loss": log_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss_epoch": loss_epoch / num_train_elems,
                    "epoch": epoch,
                }
                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_update_step)

if __name__ == "__main__":
    main()
