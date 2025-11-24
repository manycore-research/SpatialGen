import warnings

warnings.filterwarnings("ignore")  # ignore all warnings

from typing import *

import os
import argparse
import logging
import time
from copy import deepcopy
import json
import gc
import shutil

import accelerate
import torch
import torchvision
import numpy as np
import open3d as o3d
from einops import rearrange
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, AutoencoderKL, AutoencoderTiny
from diffusers import FluxControlNetModel, FluxControlPipeline

from src.options import opt_dict, Options

from src.data import ExampleDataset
from src.models.pose_adapter import RayMapEncoder

import src.utils.util as util
from src.utils.misc import worker_init_fn, todevice
from src.utils.vis_util import save_color_depth_image
from src.utils.typing import *
from src.utils.pcl_ops import rgbd_to_pointcloud, descale_depth
from diffusers_spatialgen import SpatialGenDiffusionPipeline
from diffusers_spatialgen import UNetMVMM2DConditionModel

from src.utils.traj_ops import interp_poses
from src.utils.torch3d_utils import torch3d_rasterize_points
from src.utils.depth_aligner import SmoothDepthAligner

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_autotune"

logger = logging.getLogger(__name__)


def compose_fixed_view_indices(opt: Options, device: torch.device, num_in_views: int = 1, num_sample_views: int = 2):

    num_out_views = num_sample_views - num_in_views
    num_tasks = len(opt.prediction_types) + 2 if opt.use_layout_prior else len(opt.prediction_types)

    view_indices = torch.arange(num_sample_views)[None, :]  # (1, T)
    input_indices = view_indices[:, :num_in_views]  # (1, T_in)
    target_indices = view_indices[:, num_in_views:]  # (1, T_out)

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

        pred_target_indices = torch.cat(
            [target_rgb_indices, target_depth_indices, target_normal_indices, target_semantic_indices], dim=1
        )
        pred_target_indices = rearrange(pred_target_indices, "B No -> (B No)")

        input_view_indices = rearrange(input_indices, "B Ni -> (B Ni)")
        target_view_indices = rearrange(target_indices, "B No -> (B No)")

        prediction_indices = torch.cat(
            [pred_target_indices, input_depth_indices, input_normal_indices, input_semantic_indices], dim=0
        )
        condition_indices = (
            torch.cat(
                [
                    input_rgb_indices,
                    input_layout_sem_indices,
                    target_layout_sem_indices,
                    input_layout_depth_indices,
                    target_layout_depth_indices,
                ],
                dim=0,
            )
            if opt.use_layout_prior
            else input_rgb_indices
        )

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

        prediction_indices = torch.cat([pred_target_indices, input_depth_indices, input_sem_indices], dim=0)
        condition_indices = (
            torch.cat(
                [
                    input_rgb_indices,
                    input_layout_sem_indices,
                    target_layout_sem_indices,
                    input_layout_depth_indices,
                    target_layout_depth_indices,
                ],
                dim=0,
            )
            if opt.use_layout_prior
            else input_rgb_indices
        )

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
        condition_indices = (
            torch.cat(
                [
                    input_rgb_indices,
                    input_layout_sem_indices,
                    target_layout_sem_indices,
                    input_layout_depth_indices,
                    target_layout_depth_indices,
                ],
                dim=0,
            )
            if opt.use_layout_prior
            else input_rgb_indices
        )

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
        condition_indices = (
            torch.cat(
                [
                    input_rgb_indices,
                    input_layout_sem_indices,
                    target_layout_sem_indices,
                    input_layout_depth_indices,
                    target_layout_depth_indices,
                ],
                dim=0,
            )
            if opt.use_layout_prior
            else input_rgb_indices
        )
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
        condition_indices = (
            torch.cat(
                [
                    input_rgb_indices,
                    input_layout_sem_indices,
                    target_layout_sem_indices,
                    input_layout_depth_indices,
                    target_layout_depth_indices,
                ],
                dim=0,
            )
            if opt.use_layout_prior
            else input_rgb_indices
        )
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
        condition_indices = (
            torch.cat(
                [
                    input_rgb_indices,
                    input_layout_sem_indices,
                    target_layout_sem_indices,
                    input_layout_depth_indices,
                    target_layout_depth_indices,
                ],
                dim=0,
            )
            if opt.use_layout_prior
            else input_rgb_indices
        )
    else:
        raise ValueError(f"{opt.prediction_types} is not supported")

    input_rgb_indices = input_rgb_indices.to(device).to(torch.int32)
    condition_indices = condition_indices.to(device).to(torch.int32)
    input_view_indices = input_view_indices.to(device).to(torch.int32)
    target_view_indices = target_view_indices.to(device).to(torch.int32)
    prediction_indices = prediction_indices.to(device).to(torch.int32)
    return input_rgb_indices, condition_indices, input_view_indices, target_view_indices, prediction_indices


def get_warpped_images_for_out_views(
    pipeline: SpatialGenDiffusionPipeline,
    input_images: Float[Tensor, "BNt 3 H W"],
    input_indices: Float[Tensor, "BNt"],
    input_rgb_indices: Float[Tensor, "BNt"],
    target_indices: Float[Tensor, "BNt"],
    condition_indices: Float[Tensor, "BNt"],
    output_indices: Float[Tensor, "BNt"],
    input_rays: Float[Tensor, "BNt 6 H W"],
    target_rays: Float[Tensor, "BNt 6 H W"],
    task_embeddings: Float[Tensor, "BNt 4"],
    warpped_target_images: Float[Tensor, "BNt 3 H W"],
    weight_dtype: torch.dtype,
    T_in: int,
    T_out: int,
    guidance_scale: List[float],
    num_inference_steps: int = 50,
    generator: torch.Generator = None,
    output_type: str = "numpy",
    num_tasks: int = 2,
    batch_data: Dict[str, Tensor] = None,
    target_view_poses: Float[Tensor, "B No 4 4"] = None,
    prediction_types: List[str] = ["rgb", "depth"],
    cond_input_layout_sem_images: Float[Tensor, "Bt 3 H W"] = None,
    cond_target_layout_sem_images: Float[Tensor, "Bt 3 H W"] = None,
    cond_input_layout_depth_images: Float[Tensor, "Bt 3 H W"] = None,
    cond_target_layout_depth_images: Float[Tensor, "Bt 3 H W"] = None,
    opt: Options = None,
    debug_dir: str = "",
):
    """Generate warpped images for the target views using the input images and depth maps.
    Args:

    Returns:

    """
    os.makedirs(debug_dir, exist_ok=True)
    logger.info(f"Do depth map prediction for input views")
    h, w = input_images.shape[2:]
    pred_results = pipeline(
        input_imgs=input_images,
        prompt_imgs=input_images,
        input_indices=input_indices,  # (2B x T_in)
        input_rgb_indices=input_rgb_indices,  # (B x T_in)
        target_indices=target_indices,  # (2B x T_out)
        condition_indices=condition_indices,  # (B x T_in + 2B x T_out)
        output_indices=output_indices,  # (2B x T_out + T_in)
        input_rays=input_rays,
        target_rays=target_rays,
        task_embeddings=task_embeddings,
        warpped_target_rgbs=warpped_target_images,
        torch_dtype=weight_dtype,
        height=h,
        width=w,
        T_in=T_in,
        T_out=T_out,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type=output_type,
        num_tasks=num_tasks,
        cond_input_layout_sem_images=cond_input_layout_sem_images,
        cond_target_layout_sem_images=cond_target_layout_sem_images,
        cond_input_layout_depth_images=cond_input_layout_depth_images,
        cond_target_layout_depth_images=cond_target_layout_depth_images,
    )
    pred_images = pred_results.images  # (3B x T_out + 2B x T_in, H, W, 3)
    pred_in_depth_confs = pred_results.input_depths_confi_maps  # (B x T_in, H, W, 1)
    pred_tar_depth_confs = pred_results.target_depths_confi_maps  # (B x T_out, H, W, 1)

    assert ["rgb", "depth", "semantic"] == prediction_types, "Only RGB-D-Semantic task is supported"
    num_target_rgbs, num_target_depths, num_target_sems, num_input_depths, num_input_sems = (
        T_out,
        T_out,
        T_out,
        T_in,
        T_in,
    )
    rgb_idx = num_target_rgbs
    depth_idx = num_target_rgbs + num_target_depths
    sem_idx = num_target_rgbs + num_target_depths + num_target_sems
    in_depth_idx = sem_idx + num_input_depths
    in_sem_idx = in_depth_idx + num_input_sems
    fake_tar_rgbs = torch.from_numpy(pred_images[:rgb_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
    if opt.use_scene_coord_map:
        # actual scene coord map
        fake_in_depths = (
            torch.from_numpy(pred_images[sem_idx:in_depth_idx, :, :, :] * 2.0 - 1.0)
            .permute(0, 3, 1, 2)
            .to(input_images)
        )  # (BxT)x3xhxw
        # filter the depth map with the confidence map
        assert pred_in_depth_confs is not None
        pred_in_depth_confs = torch.from_numpy(pred_in_depth_confs).permute(0, 3, 1, 2).to(input_images)  # (BxT)x1xhxw
        logger.info(
            f"pred_in_depth_confs min: {pred_in_depth_confs.min()}, median: {pred_in_depth_confs.median()}, max: {pred_in_depth_confs.max()}"
        )
        torchvision.utils.save_image(pred_in_depth_confs, f"{debug_dir}/pred_in_depth_confs.png", normalize=True)
        pred_in_depth_masks = (pred_in_depth_confs > 6.0).to(input_images)
        fake_in_depths = fake_in_depths * pred_in_depth_masks
    else:
        fake_in_depths = (
            torch.from_numpy(pred_images[sem_idx:in_depth_idx, :, :, 0:1] * 2.0 - 1.0)
            .permute(0, 3, 1, 2)
            .to(input_images)
        )  # (BxT)x1xhxw

    # render the warpped images at the target views
    min_depth = batch_data["depth_min"][0:1].to(weight_dtype)  # B,1
    max_depth = batch_data["depth_max"][0:1].to(weight_dtype)  # B,1
    scene_scale = batch_data["scene_scale"][0:1].to(weight_dtype)  # B,1
    intrinsics = batch_data["intrinsic"][0:1].to(weight_dtype)  # B,3,3
    # warp the depth maps to target views
    colors = rearrange(input_images * 0.5 + 0.5, "(b t) c h w -> b t c h w", t=T_in)
    depths = rearrange(fake_in_depths * 0.5 + 0.5, "(b t) c h w -> b t c h w", t=T_in)
    depths = (
        descale_depth(depths, min_depth[:, None, None, None, None], max_depth[:, None, None, None, None])
        * scene_scale[:, None, None, None, None]
    )
    batch_input_points: Float[Tensor, "B Np 3"] = rearrange(depths, "B Ni C H W -> B (Ni H W) C", Ni=T_in, H=h, W=w)
    batch_input_colors: Float[Tensor, "B 3 Np"] = rearrange(colors, "B Ni C H W -> B (Ni H W) C", Ni=T_in, H=h, W=w)

    pointcloud = torch.cat([batch_input_points, batch_input_colors], dim=-1)  # B, Np, 6
    pointcloud: Float[Tensor, "BNp 6"] = rearrange(pointcloud, "B Np C -> (B Np) C")
    projected_tar_imgs, projected_tar_depths = torch3d_rasterize_points(
        cv_cam_poses_c2w=rearrange(target_view_poses.float(), "b t h w -> (b t) h w"),
        in_pointcloud=pointcloud.float(),
        intrinsic=intrinsics[0].float(),
        image_width=w,
        image_height=h,
        point_radius=0.01,
        device=input_images.device,
    )
    warpped_target_images = projected_tar_imgs * 2.0 - 1.0

    # convert depth into colorful depth
    vis_pre_in_depths = torch.from_numpy(pred_images[sem_idx:in_depth_idx, :, :, :]).permute(0, 3, 1, 2)
    vis_pre_in_depths = torchvision.utils.make_grid(vis_pre_in_depths, nrow=1)
    torchvision.utils.save_image(vis_pre_in_depths, f"{debug_dir}/pred1_in_depths.png")

    return warpped_target_images.to(input_images).clamp(-1.0, 1.0)


def align_and_save_pointcloud(
    input_images: Float[Tensor, "BNt 3 H W"],
    input_depths: Float[Tensor, "BNt 1 H W"],
    output_images: Float[Tensor, "BNt 3 H W"],
    output_depths: Float[Tensor, "BNt 1 H W"],
    poses_input: Float[Tensor, "BNt 4 4"],
    poses_output: Float[Tensor, "BNt 4 4"],
    intrinsics: Float[Tensor, "3 3"],
    min_depth: Float[Tensor, "BNt "],
    max_depth: Float[Tensor, "BNt "],
    scene_scale: Float[Tensor, "BNt "],
    output_folder: str = "./debug_output",
    input_sem_images: Float[Tensor, "BNt 3 H W"] = None,
    output_sem_images: Float[Tensor, "BNt 3 H W"] = None,
    return_data_dict: Dict[str, Any] = None,
    is_gt: bool = False,
    only_return_target_ply: bool = False,
    reference_renderings: Optional[Tuple] = None,
    input_depth_masks: Optional[Float[Tensor, "BNt 1 H W"]] = None,
    output_depth_masks: Optional[Float[Tensor, "BNt 1 H W"]] = None,
) -> o3d.geometry.PointCloud:
    """
    align and save input and output point cloud

    Args:
        input_images: [BxT, C, H, W], range [-1, 1]
        input_depths: [BxT, 1, H, W], range [-1, 1]
        output_images: [BxT, C, H, W], range [-1, 1]
        output_depths: [BxT, 1, H, W], range [-1, 1]
        poses_input: [BxT, 4, 4], input view poses
        poses_output: [BxT, 4, 4], output view poses
        intrinsics: [3, 3], camera intrinsics
        min_depth: [B], min depth
        max_depth: [B], max depth
        scene_scale: [B], scene scale
        output_folder: str, folder to save the point cloud
        return_data_dict: dict, data dict to save the point cloud
        is_gt: bool, whether the input images are ground truth
        only_return_target_ply: bool, whether to return only the target point cloud
        reference_renderings: tuple, reference renderings for depth alignment
    Returns:
        o3d.geometry.PointCloud: aligned point cloud
    """
    return_data_dict = {} if return_data_dict is None else return_data_dict
    os.makedirs(output_folder, exist_ok=True)

    prefix_str = "gt_" if is_gt else "pred_"

    input_rgbs = (input_images.cpu() + 1.0) / 2.0
    inputs = torchvision.utils.make_grid(input_rgbs, nrow=1)
    torchvision.utils.save_image(inputs, os.path.join(output_folder, "input_rgbs.png"))
    output_rgbs = (output_images + 1.0) / 2.0
    outputs = torchvision.utils.make_grid(torch.cat([output_rgbs]), nrow=1)
    torchvision.utils.save_image(outputs, os.path.join(output_folder, f"{prefix_str}tar_rgbs.png"))

    input_rgbs = input_rgbs.clone().permute(0, 2, 3, 1).cpu().numpy() * 255.0
    output_rgbs = output_rgbs.clone().permute(0, 2, 3, 1).cpu().numpy() * 255.0
    input_semantics = (input_sem_images.cpu() * 0.5 + 0.5).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    output_semantics = (output_sem_images.cpu() * 0.5 + 0.5).permute(0, 2, 3, 1).cpu().numpy() * 255.0

    input_depths = descale_depth((input_depths.clone() + 1.0) / 2.0, min_depth, max_depth) * scene_scale
    output_depths = descale_depth((output_depths.clone() + 1.0) / 2.0, min_depth, max_depth) * scene_scale
    filtered_input_depths = input_depths.permute(0, 2, 3, 1).cpu().numpy()
    filtered_output_depths = output_depths.permute(0, 2, 3, 1).cpu().numpy()

    input_rgb_ply = o3d.geometry.PointCloud()
    total_rgb_ply = o3d.geometry.PointCloud()
    input_semantic_ply = o3d.geometry.PointCloud()
    total_semantic_ply = o3d.geometry.PointCloud()
    img_height, img_width = input_rgbs.shape[1], input_rgbs.shape[2]

    depth_aligner = SmoothDepthAligner()

    input_rgb_ply = o3d.geometry.PointCloud()
    total_rgb_ply = o3d.geometry.PointCloud()
    aligned_rgb_ply = o3d.geometry.PointCloud()
    img_height, img_width = input_rgbs.shape[1], input_rgbs.shape[2]

    save_data_dict = {}
    in_rgbs, in_semantics, input_depths, input_poses = [], [], [], []
    target_rgbs, target_semantics, target_depths, target_poses = [], [], [], []
    input_points, input_colors = [], []
    target_points, target_colors = [], []
    for id, (rgb, semantic, depth, c2w_pose) in enumerate(
        zip(input_rgbs, input_semantics, filtered_input_depths, poses_input)
    ):

        scene_coord_maps = deepcopy(depth).reshape(-1, 3)
        # calculate the depth map
        w2c_pose = np.linalg.inv(c2w_pose.detach().cpu().numpy())
        depth = (w2c_pose[:3, :3] @ scene_coord_maps.T + w2c_pose[:3, 3:4]).T.reshape(img_height, img_width, 3)
        save_color_depth_image(
            depth[None, :, :, 2], cmap="jet", output_path=os.path.join(output_folder, f"pred_in_depth_{id}.png")
        )
        if input_depth_masks is not None:
            input_mask = input_depth_masks[id].cpu().numpy()
            depth_map = depth[:, :, 2:3]
            scene_coord_maps = scene_coord_maps * input_mask.reshape(-1, 1)
        else:
            depth_map = depth[:, :, 2:3]

        o3d_recon_ply = o3d.geometry.PointCloud()
        o3d_recon_ply.points = o3d.utility.Vector3dVector(scene_coord_maps)
        o3d_recon_ply.colors = o3d.utility.Vector3dVector((rgb / 255.0).reshape(-1, 3))

        o3d_sem_ply = o3d.geometry.PointCloud()
        o3d_sem_ply.points = o3d.utility.Vector3dVector(scene_coord_maps)
        o3d_sem_ply.colors = o3d.utility.Vector3dVector((semantic / 255.0).reshape(-1, 3))

        input_rgb_ply += o3d_recon_ply
        input_semantic_ply += o3d_sem_ply

        in_rgbs.append(rgb)
        in_semantics.append(semantic)
        input_depths.append(depth_map[:, :, 0])
        input_poses.append(c2w_pose.detach().cpu().numpy())
        input_points.append(np.asarray(o3d_recon_ply.points))
        input_colors.append(np.asarray(o3d_recon_ply.colors))

    if not only_return_target_ply:
        total_rgb_ply += input_rgb_ply
        total_semantic_ply += input_semantic_ply

    for id, (rgb, semantic, depth, c2w_pose) in enumerate(
        zip(output_rgbs, output_semantics, filtered_output_depths, poses_output)
    ):
        scene_coord_maps = deepcopy(depth).reshape(-1, 3)
        # calculate the depth map
        w2c_pose = np.linalg.inv(c2w_pose.detach().cpu().numpy())
        depth = (w2c_pose[:3, :3] @ scene_coord_maps.T + w2c_pose[:3, 3:4]).T.reshape(img_height, img_width, 3)
        save_color_depth_image(
            depth[None, :, :, 2], cmap="jet", output_path=os.path.join(output_folder, f"pred_tar_depth_{id}.png")
        )
        if output_depth_masks is not None:
            output_mask = output_depth_masks[id].cpu().numpy()
            depth_map = depth[:, :, 2:3]
            scene_coord_maps = scene_coord_maps * output_mask.reshape(-1, 1)
        else:
            depth_map = depth[:, :, 2:3]

        o3d_recon_ply = o3d.geometry.PointCloud()
        o3d_recon_ply.points = o3d.utility.Vector3dVector(scene_coord_maps)
        o3d_recon_ply.colors = o3d.utility.Vector3dVector((rgb / 255.0).reshape(-1, 3))

        o3d_sem_ply = o3d.geometry.PointCloud()
        o3d_sem_ply.points = o3d.utility.Vector3dVector(scene_coord_maps)
        o3d_sem_ply.colors = o3d.utility.Vector3dVector((semantic / 255.0).reshape(-1, 3))

        total_semantic_ply += o3d_sem_ply
        total_rgb_ply += o3d_recon_ply

        target_rgbs.append(rgb)
        target_semantics.append(semantic)
        target_depths.append(depth_map[:, :, 0])
        target_poses.append(c2w_pose.detach().cpu().numpy())
        target_points.append(np.asarray(o3d_recon_ply.points))
        target_colors.append(np.asarray(o3d_recon_ply.colors))

        # DOING depth alignment
        if reference_renderings is not None:
            rendered_tar_rgbs, rendered_tar_depths, rendered_masks = reference_renderings
            rendered_tar_rgbs = rendered_tar_rgbs.permute(0, 2, 3, 1)[id : id + 1]
            rendered_tar_depths = rendered_tar_depths.permute(0, 2, 3, 1)[id : id + 1]
            rendered_masks = rendered_masks.permute(0, 2, 3, 1)[id : id + 1]
            reference_depth: Float[np.ndarray, "H W"] = rendered_tar_depths.cpu().numpy()[0, :, :, 0]
            rendered_rgbs: Float[np.ndarray, "H W"] = rendered_tar_rgbs.cpu().numpy()[0, ...]
            reference_mask: Float[np.ndarray, "H W"] = rendered_masks.cpu().numpy()[0, :, :, 0]
            # align the predicted depth with the rendered depth
            aligned_depth_map = depth_aligner.align_depth(
                reference_dpt=reference_depth, new_dpt=depth_map[..., 0], reference_msk=reference_mask
            )
            aligned_depth_map = aligned_depth_map[:, :, np.newaxis]
            # from matplotlib import pyplot as plt
            # fig, axs = plt.subplots(1, 5, figsize=(15, 5))
            # axs[0].imshow(rendered_rgbs)
            # axs[1].imshow(reference_depth)
            # axs[2].imshow(reference_mask)
            # axs[3].imshow(aligned_depth_map[..., 0])
            # axs[4].imshow(depth_map[..., 0])
            # plt.savefig(os.path.join(output_folder, f"rendered_tar_{id}.png"))
            # plt.close()
            aligned_ply = rgbd_to_pointcloud(
                rgb_image=rgb,
                depth_image=aligned_depth_map[:, :, 0],
                c2w_pose=c2w_pose.detach().cpu().numpy(),
                depth_scale=1.0,
                intrinsic_mat=intrinsics.cpu().numpy(),
            )
            aligned_rgb_ply += aligned_ply

    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}input_rgb.ply"), input_rgb_ply)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}total_rgb.ply"), total_rgb_ply)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}input_semantic.ply"), input_semantic_ply)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}total_semantic.ply"), total_semantic_ply)
    if aligned_rgb_ply.has_points():
        o3d.io.write_point_cloud(os.path.join(output_folder, f"{prefix_str}aligned_rgb.ply"), aligned_rgb_ply)

    # save data into a npz file
    save_data_dict["input_rgbs"] = in_rgbs
    save_data_dict["input_semantics"] = in_semantics
    save_data_dict["input_depths"] = input_depths
    save_data_dict["input_poses"] = input_poses
    save_data_dict["input_points"] = input_points
    save_data_dict["input_colors"] = input_colors
    save_data_dict["target_rgbs"] = target_rgbs
    save_data_dict["target_semantics"] = target_semantics
    save_data_dict["target_depths"] = target_depths
    save_data_dict["target_poses"] = target_poses
    save_data_dict["target_points"] = target_points
    save_data_dict["target_colors"] = target_colors
    save_data_dict["intrinsic"] = intrinsics.cpu().numpy()
    return total_rgb_ply, save_data_dict, total_semantic_ply


def vis_all_imgs(
    in_view_rgbs: Float[Tensor, "BNi 3 H W"],
    tar_view_rgbs: Float[Tensor, "BNo 3 H W"],
    in_view_lay_deps: Float[Tensor, "BNi 3 H W"],
    tar_view_lay_deps: Float[Tensor, "BNo 3 H W"],
    in_view_lay_sems: Float[Tensor, "BNi 3 H W"],
    tar_view_lay_sems: Float[Tensor, "BNo 3 H W"],
    in_view_depths: Float[Tensor, "BNi 1 H W"],
    tar_view_depths: Float[Tensor, "BNo 1 H W"],
    in_view_sems: Float[Tensor, "BNi 3 H W"],
    tar_view_sems: Float[Tensor, "BNo 3 H W"],
    output_prefix: str = "all_preds",
    output_folder: str = "./",
):
    vis_in_view_rgbs = in_view_rgbs.cpu() * 0.5 + 0.5
    vis_tar_view_rgbs = tar_view_rgbs.cpu() * 0.5 + 0.5
    if in_view_lay_deps is not None:
        vis_in_view_lay_deps = in_view_lay_deps.cpu()
    if tar_view_lay_deps is not None:
        vis_tar_view_lay_deps = tar_view_lay_deps.cpu()
    if in_view_lay_sems is not None:
        vis_in_view_lay_sems = in_view_lay_sems.cpu()
    if tar_view_lay_sems is not None:
        vis_tar_view_lay_sems = tar_view_lay_sems.cpu()
    vis_in_view_deps = in_view_depths.cpu() * 0.5 + 0.5
    vis_tar_view_deps = tar_view_depths.cpu() * 0.5 + 0.5
    vis_in_view_sems = in_view_sems.cpu()
    vis_tar_view_sems = tar_view_sems.cpu()

    vis_all_view_rgbs = torch.cat([vis_in_view_rgbs, vis_tar_view_rgbs], dim=0)
    for i in range(vis_all_view_rgbs.shape[0]):
        rgb = vis_all_view_rgbs[i]
        torchvision.utils.save_image(rgb, os.path.join(output_folder, f"rgb_{i}.png"))
    vis_all_view_rgbs = torchvision.utils.make_grid(vis_all_view_rgbs)
    if in_view_lay_deps is not None and tar_view_lay_deps is not None:
        vis_all_view_lay_deps = torch.cat([vis_in_view_lay_deps, vis_tar_view_lay_deps], dim=0)
        for i in range(vis_all_view_lay_deps.shape[0]):
            rgb = vis_all_view_lay_deps[i]
            torchvision.utils.save_image(rgb, os.path.join(output_folder, f"lay_dep_{i}.png"))
        vis_all_view_lay_deps = torchvision.utils.make_grid(vis_all_view_lay_deps)
    if in_view_lay_sems is not None and tar_view_lay_sems is not None:
        vis_all_view_lay_sems = torch.cat([vis_in_view_lay_sems, vis_tar_view_lay_sems], dim=0)
        for i in range(vis_all_view_lay_sems.shape[0]):
            rgb = vis_all_view_lay_sems[i]
            torchvision.utils.save_image(rgb, os.path.join(output_folder, f"lay_sem_{i}.png"))
        vis_all_view_lay_sems = torchvision.utils.make_grid(vis_all_view_lay_sems)

    vis_all_view_deps = torch.cat([vis_in_view_deps, vis_tar_view_deps], dim=0)
    for i in range(vis_all_view_deps.shape[0]):
        rgb = vis_all_view_deps[i]
        torchvision.utils.save_image(rgb, os.path.join(output_folder, f"dep_{i}.png"))
    vis_all_view_deps = torchvision.utils.make_grid(vis_all_view_deps)

    vis_all_view_sems = torch.cat([vis_in_view_sems, vis_tar_view_sems], dim=0)
    for i in range(vis_all_view_sems.shape[0]):
        rgb = vis_all_view_sems[i]
        torchvision.utils.save_image(rgb, os.path.join(output_folder, f"sem_{i}.png"))
    vis_all_view_sems = torchvision.utils.make_grid(vis_all_view_sems)

    if (
        in_view_lay_deps is not None
        and tar_view_lay_deps is not None
        and in_view_lay_sems is not None
        and tar_view_lay_sems is not None
    ):
        all_imgs = torch.cat(
            [vis_all_view_lay_deps, vis_all_view_lay_sems, vis_all_view_rgbs, vis_all_view_deps, vis_all_view_sems],
            dim=-2,
        )
    else:
        all_imgs = torch.cat([vis_all_view_rgbs, vis_all_view_deps, vis_all_view_sems], dim=-2)
    logger.info(f"all_imgs shape: {all_imgs.shape}")
    torchvision.utils.save_image(all_imgs, os.path.join(output_folder, f"{output_prefix}.png"))


@torch.no_grad()
def model_inference(
    args,
    round_idx: int,
    opt: Options,
    pipeline: SpatialGenDiffusionPipeline,
    weight_dtype: torch.dtype,
    in_view_ids: np.array,
    tar_view_ids: np.array,
    num_tasks: int = 2,
    generator: torch.Generator = None,
    batch: Dict[str, Tensor] = None,
    in_view_rgbs: Float[Tensor, "B Ni 3 H W"] = None,
    in_view_scms: Float[Tensor, "B Ni 3 H W"] = None,
    in_view_sems: Float[Tensor, "B Ni 3 H W"] = None,
    scene_ply: Float[Tensor, "N 6"] = None,
    scm_conf_thresh: float = 5.0,
    output_dir: str = "",
) -> Tuple[Float[Tensor, "N 6"], Float[Tensor, "Bn 3 H W"]]:
    """
    Inference the 8-view MVD model in a single iteration:
    1. Given the in_view_rgbs, generate the target views at tar_view_ids;
    2. For the 8-view tuple, interpolate the intermediate views for each 2 adjacent pairs.

    Args:
        args: command line arguments
        round_idx: the current round index
        opt: options for the model
        pipeline: the model pipeline
        weight_dtype: the data type of the model weights
        in_view_ids: the input view indices
        tar_view_ids: the target view indices
        num_tasks: the number of multi-modal tasks
        generator: the random number generator
        batch: the batch data
        in_view_rgbs: the input view RGB images, [-1, 1]
        in_view_scms: the input view scene point clouds, [-1, 1]
        in_view_sems: the input view semantic maps, [-1, 1]
        scene_ply: the global scene point cloud
        output_dir: the output directory for saving results
    Returns:
        recons_ply: the generated point cloud;
        pred_images: the generated images, [-1, 1]
    """
    num_in_views = len(in_view_ids)
    num_tar_views = len(tar_view_ids)
    selected_view_ids = torch.tensor(list(in_view_ids) + list(tar_view_ids), dtype=torch.int32)  # (T_in + T_out)

    room_uid = batch["room_uid"][0].replace("/", "_")
    output_folder = output_dir
    os.makedirs(output_folder, exist_ok=True)

    all_rgbs = torch.cat([batch["image_input"][0:1], batch["image_target"][0:1]], dim=1).to(
        dtype=weight_dtype
    )  # B,N,3,H,W
    if opt.dataset_names == ["spatialgen"]:
        all_depths = torch.cat([batch["depth_input"][0:1], batch["depth_target"][0:1]], dim=1).to(
            dtype=weight_dtype
        )  # B,N,3,H,W
        all_sems = torch.cat([batch["semantic_input"][0:1], batch["semantic_target"][0:1]], dim=1).to(
            weight_dtype
        )  # B,N,3,H,W
    if opt.use_layout_prior:
        all_layout_sem_images = torch.cat(
            [batch["semantic_layout_input"][0:1], batch["semantic_layout_target"][0:1]], dim=1
        ).to(
            weight_dtype
        )  # B,N,3,H,W
        all_layout_dep_images = torch.cat(
            [batch["depth_layout_input"][0:1], batch["depth_layout_target"][0:1]], dim=1
        ).to(
            weight_dtype
        )  # B,N,3,H,W
    else:
        # raise NotImplementedError("WO-Layout prior is not supported yet.")
        all_layout_sem_images = None
        all_layout_dep_images = None
    all_rays = torch.cat([batch["plucker_rays_input"][0:1], batch["plucker_rays_target"][0:1]], dim=1).to(
        dtype=weight_dtype
    )
    all_c2w_poses = torch.cat([batch["pose_in"][0:1], batch["pose_out"][0:1]], dim=1)  # B,N,4,4
    all_c2w_metric_poses = torch.cat(
        [batch["pose_metric_input"][0:1], batch["pose_metric_target"][0:1]], dim=1
    )  # B,N,4,4
    in_view_metric_poses = all_c2w_metric_poses[0, in_view_ids]  # T_in,4,4
    tar_view_metric_poses = all_c2w_metric_poses[0, tar_view_ids]  # T_out,4,4

    ### 1. generate the 8-View tuples, according to RGB-Scm-Sem images
    intrinsic_mat = batch["intrinsic"][0].cpu()
    in_view_poses = all_c2w_poses[0:1, in_view_ids]  # B,T_in,4,4
    tar_view_poses = all_c2w_poses[0:1, tar_view_ids]  # B,T_out,4,4
    if in_view_rgbs is None:
        in_view_rgbs = rearrange(all_rgbs[0:1, in_view_ids], "b t c h w -> (b t) c h w", t=num_in_views)
    else:
        in_view_rgbs = rearrange(in_view_rgbs, "b t c h w -> (b t) c h w", t=num_in_views)
    tar_view_rgbs = rearrange(all_rgbs[0:1, tar_view_ids], "b t c h w -> (b t) c h w", t=num_tar_views)
    if opt.dataset_names == ["spatialgen"]:
        tar_view_depths = rearrange(all_depths[0:1, tar_view_ids], "b t c h w -> (b t) c h w", t=num_tar_views)
        tar_view_sems = rearrange(all_sems[0:1, tar_view_ids], "b t c h w -> (b t) c h w", t=num_tar_views)
    in_view_rays = torch.cat([all_rays[0:1, in_view_ids]] * num_tasks)  # B,1,6,h,w
    tar_view_rays = torch.cat([all_rays[0:1, tar_view_ids]] * num_tasks)  # B,7,6,h,w
    if opt.use_layout_prior:
        in_view_layout_deps = rearrange(
            all_layout_dep_images[0:1, in_view_ids], "b t c h w -> (b t) c h w", t=num_in_views
        )
        in_view_layout_sems = rearrange(
            all_layout_sem_images[0:1, in_view_ids], "b t c h w -> (b t) c h w", t=num_in_views
        )
        tar_view_layout_sems = rearrange(
            all_layout_sem_images[0:1, tar_view_ids], "b t c h w -> (b t) c h w", t=num_tar_views
        )
        tar_view_layout_deps = rearrange(
            all_layout_dep_images[0:1, tar_view_ids], "b t c h w -> (b t) c h w", t=num_tar_views
        )
        if in_view_scms is not None:
            in_view_layout_deps = rearrange(in_view_scms, "b t c h w -> (b t) c h w", t=num_in_views)
        if in_view_sems is not None:
            in_view_layout_sems = rearrange(in_view_sems, "b t c h w -> (b t) c h w", t=num_in_views)
    else:
        in_view_layout_deps = None
        in_view_layout_sems = None
        tar_view_layout_sems = None
        tar_view_layout_deps = None

    # prepare task embedding
    selected_view_ids = selected_view_ids.to(all_rgbs.device)
    task_embeddings = torch.cat(
        [
            batch["color_task_embeddings"][0:1, selected_view_ids, :],
            batch["depth_task_embeddings"][0:1, selected_view_ids, :],
            batch["semantic_task_embeddings"][0:1, selected_view_ids, :],
        ],
        dim=1,
    ).to(
        dtype=weight_dtype
    )  # B, 3*2, 4
    if opt.use_layout_prior:
        task_embeddings = torch.cat(
            [
                task_embeddings,
                batch["layout_sem_task_embeddings"][0:1, selected_view_ids, :],
                batch["layout_depth_task_embeddings"][0:1, selected_view_ids, :],
            ],
            dim=1,
        )  # B, 5*2, 4
    task_embeddings = rearrange(task_embeddings, "b t c -> (b t) c").contiguous()  # B*num_tasks*(T_in+T_out), 4

    input_rgb_indices, condition_indices, input_view_indices, target_view_indices, prediction_indices = (
        compose_fixed_view_indices(
            opt, device=all_rgbs.device, num_in_views=num_in_views, num_sample_views=(num_in_views + num_tar_views)
        )
    )

    guidance_scale = args.guidance_scale
    h, w = in_view_rgbs.shape[2:]

    if opt.input_concat_warpped_image:
        if scene_ply is None:
            # generate warpped images for target views
            warpped_target_images = get_warpped_images_for_out_views(
                pipeline,
                input_images=in_view_rgbs,
                input_indices=input_view_indices,
                input_rgb_indices=input_rgb_indices,
                target_indices=target_view_indices,
                condition_indices=condition_indices,  # (B x T_in + 2B x (T_in + T_out))
                output_indices=prediction_indices,
                input_rays=in_view_rays,
                target_rays=tar_view_rays,
                task_embeddings=task_embeddings,
                warpped_target_images=torch.full_like(tar_view_rgbs, -1),
                weight_dtype=weight_dtype,
                T_in=num_in_views,
                T_out=num_tar_views,
                guidance_scale=guidance_scale,
                num_inference_steps=opt.num_inference_steps,
                generator=generator,
                output_type="numpy",
                num_tasks=num_tasks,
                batch_data=batch,
                target_view_poses=tar_view_poses,
                prediction_types=opt.prediction_types,
                opt=opt,
                cond_input_layout_sem_images=in_view_layout_sems,
                cond_target_layout_sem_images=tar_view_layout_sems,
                cond_input_layout_depth_images=in_view_layout_deps,
                cond_target_layout_depth_images=tar_view_layout_deps,
                debug_dir=output_folder,
            )
        else:
            # render warpped image using the scene ply
            projected_tar_imgs, projected_tar_depths = torch3d_rasterize_points(
                cv_cam_poses_c2w=rearrange(tar_view_poses.float(), "b t h w -> (b t) h w"),
                in_pointcloud=scene_ply.float(),
                intrinsic=intrinsic_mat.float(),
                image_width=w,
                image_height=h,
                point_radius=0.01,
                device=all_rgbs.device,
            )
            warpped_target_images = (projected_tar_imgs * 2.0 - 1.0).clamp(-1.0, 1.0)
    else:
        warpped_target_images = None

    with torch.autocast("cuda", weight_dtype):
        pred_results = pipeline(
            input_imgs=in_view_rgbs,
            prompt_imgs=in_view_rgbs,
            input_indices=input_view_indices,  # (2B x T_in)
            input_rgb_indices=input_rgb_indices,  # (B x T_in)
            condition_indices=condition_indices,  # (B x T_in + 2B x (T_in + T_out))
            target_indices=target_view_indices,  # (2B x T_out)
            output_indices=prediction_indices,  # (2B x T_out + T_in)
            input_rays=in_view_rays,
            target_rays=tar_view_rays,
            task_embeddings=task_embeddings,
            warpped_target_rgbs=warpped_target_images,
            torch_dtype=weight_dtype,
            height=h,
            width=w,
            T_in=num_in_views,
            T_out=num_tar_views,
            guidance_scale=guidance_scale,
            num_inference_steps=opt.num_inference_steps,
            generator=generator,
            output_type="numpy",
            num_tasks=num_tasks,
            cond_input_layout_sem_images=in_view_layout_sems,
            cond_target_layout_sem_images=tar_view_layout_sems,
            cond_input_layout_depth_images=in_view_layout_deps,
            cond_target_layout_depth_images=tar_view_layout_deps,
        )
        pred_images = pred_results.images  # (3B x T_out + 2B x T_in, H, W, 3)
        pred_in_depth_confs = pred_results.input_depths_confi_maps  # (B x T_in, H, W, 1)
        pred_tar_depth_confs = pred_results.target_depths_confi_maps  # (B x T_out, H, W, 1)
        # save warpped target rgbs
        if opt.input_concat_warpped_image:
            warpped_target_rgbs = (warpped_target_images + 1.0) / 2.0
            warpped_target_rgbs = torchvision.utils.make_grid(warpped_target_rgbs, nrow=1)
            torchvision.utils.save_image(warpped_target_rgbs, f"{output_folder}/warpped_target_rgbs.png")
        # save layout condition images
        if opt.use_layout_prior:
            in_lay_sem_images = (in_view_layout_sems + 1.0) / 2.0
            torchvision.utils.save_image(
                torchvision.utils.make_grid(in_lay_sem_images, nrow=1), f"{output_folder}/input_layout_semantics.png"
            )

            tar_lay_sem_images = (tar_view_layout_sems + 1.0) / 2.0
            torchvision.utils.save_image(
                torchvision.utils.make_grid(tar_lay_sem_images, nrow=1), f"{output_folder}/target_layout_semantics.png"
            )

            if not opt.use_scene_coord_map:
                in_lay_depth_images = ((in_view_layout_deps + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].cpu().numpy()
                in_lay_depth_images = save_color_depth_image(
                    in_lay_depth_images, output_path=f"{output_folder}/input_layout_depths.png"
                )

                tar_lay_depth_images = (
                    ((tar_view_layout_deps + 1.0) / 2.0).permute(0, 2, 3, 1)[:, :, :, 0].cpu().numpy()
                )
                tar_lay_depth_images = save_color_depth_image(
                    tar_lay_depth_images, output_path=f"{output_folder}/target_layout_depths.png"
                )
            else:
                in_lay_depth_images = (in_view_layout_deps + 1.0) / 2.0
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(in_lay_depth_images, nrow=1), f"{output_folder}/input_layout_depths.png"
                )
                tar_lay_depth_images = (tar_view_layout_deps + 1.0) / 2.0
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(tar_lay_depth_images, nrow=1),
                    f"{output_folder}/target_layout_depths.png",
                )
        gt_tar_rgbs = torchvision.utils.make_grid(tar_view_rgbs * 0.5 + 0.5, nrow=1)
        torchvision.utils.save_image(gt_tar_rgbs, os.path.join(output_folder, "gt_tar_rgbs.png"))

        num_target_rgbs, num_target_depths, num_target_sems, num_input_depths, num_input_sems = (
            num_tar_views,
            num_tar_views,
            num_tar_views,
            num_in_views,
            num_in_views,
        )
        rgb_idx = num_target_rgbs
        depth_idx = num_target_rgbs + num_target_depths
        sem_idx = num_target_rgbs + num_target_depths + num_target_sems
        in_depth_idx = sem_idx + num_input_depths
        in_sem_idx = in_depth_idx + num_input_sems
        fake_tar_rgbs = torch.from_numpy(pred_images[:rgb_idx, :, :, :] * 2.0 - 1.0).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        if opt.use_scene_coord_map:
            # actual scene coord map
            fake_tar_depths = torch.from_numpy(pred_images[rgb_idx:depth_idx, :, :, :] * 2.0 - 1.0).permute(
                0, 3, 1, 2
            )  # (BxT)x3xhxw
            fake_in_depths = torch.from_numpy(pred_images[sem_idx:in_depth_idx, :, :, :] * 2.0 - 1.0).permute(
                0, 3, 1, 2
            )  # (BxT)x3xhxw
            torchvision.utils.save_image(
                torchvision.utils.make_grid(fake_tar_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/pred_tar_depths.png"
            )
            torchvision.utils.save_image(
                torchvision.utils.make_grid(fake_in_depths * 0.5 + 0.5, nrow=1), f"{output_folder}/pred_in_depths.png"
            )

            # filter the depth map with the confidence map
            assert pred_in_depth_confs is not None and pred_tar_depth_confs is not None
            pred_in_depth_confs = torch.from_numpy(pred_in_depth_confs).permute(0, 3, 1, 2)  # (BxT)x1xhxw
            logger.info(
                f"pred_in_depth_confs min: {pred_in_depth_confs.min()}, median: {pred_in_depth_confs.median()}, max: {pred_in_depth_confs.max()}"
            )
            torchvision.utils.save_image(
                pred_in_depth_confs, f"{output_folder}/pred_in_depth_confs.png", normalize=True
            )
            pred_in_depth_masks = (pred_in_depth_confs > scm_conf_thresh).float()
            vis_fake_in_depths = fake_in_depths
            # fake_in_depths = fake_in_depths * pred_in_depth_masks
            pred_tar_depth_confs = torch.from_numpy(pred_tar_depth_confs).permute(0, 3, 1, 2)
            logger.info(
                f"pred_tar_depth_confs min: {pred_tar_depth_confs.min()}, median: {pred_tar_depth_confs.median()}, max: {pred_tar_depth_confs.max()}"
            )
            torchvision.utils.save_image(
                pred_tar_depth_confs, f"{output_folder}/pred_tar_depth_confs.png", normalize=True
            )
            pred_tar_depth_masks = (pred_tar_depth_confs > scm_conf_thresh).float()
            vis_fake_tar_depths = fake_tar_depths
            # fake_tar_depths = fake_tar_depths * pred_tar_depth_masks
        else:
            fake_tar_depths = torch.from_numpy(pred_images[rgb_idx:depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(
                0, 3, 1, 2
            )  # (BxT)x1xhxw
            pred_tar_depth_masks = None
            fake_in_depths = torch.from_numpy(pred_images[sem_idx:in_depth_idx, :, :, 0:1] * 2.0 - 1.0).permute(
                0, 3, 1, 2
            )  # (BxT)x1xhxw
            pred_in_depth_masks = None

        fake_tar_sems = torch.from_numpy(pred_images[depth_idx:sem_idx, :, :, :]).permute(0, 3, 1, 2)  # (BxT)x3xhxw
        fake_in_sems = torch.from_numpy(pred_images[in_depth_idx:in_sem_idx, :, :, :]).permute(
            0, 3, 1, 2
        )  # (BxT)x3xhxw
        vis_fake_in_sems = torchvision.utils.make_grid(fake_in_sems, nrow=1)
        torchvision.utils.save_image(vis_fake_in_sems, os.path.join(output_folder, "pred_in_semantics.png"))
        vis_fake_tar_sems = torchvision.utils.make_grid(fake_tar_sems, nrow=1)
        torchvision.utils.save_image(vis_fake_tar_sems, os.path.join(output_folder, "pred_tar_semantics.png"))

        in_view_poses = rearrange(in_view_poses.float(), "b t c d-> (b t) c d", t=num_in_views)
        tar_view_poses = rearrange(tar_view_poses.float(), "b t c d -> (b t) c d", t=num_tar_views)
        min_depth = batch["depth_min"][0:1].cpu()  # B,1
        max_depth = batch["depth_max"][0:1].cpu()
        scene_scale = batch["scene_scale"][0:1].cpu()
        print("validating RGB-D-Sem task on room {}".format(room_uid))

        # align the generated scm with the global SCM
        recons_ply, room_infer_results, recons_sem_ply = align_and_save_pointcloud(
            input_images=in_view_rgbs.float(),
            input_depths=fake_in_depths.float(),
            output_images=fake_tar_rgbs.float(),
            output_depths=fake_tar_depths.float(),
            input_sem_images=fake_in_sems.float() * 2.0 - 1.0,
            output_sem_images=fake_tar_sems.float() * 2.0 - 1.0,
            poses_input=in_view_poses,
            poses_output=tar_view_poses,
            intrinsics=intrinsic_mat.float(),
            min_depth=min_depth.float(),
            max_depth=max_depth.float(),
            scene_scale=scene_scale.float(),
            output_folder=output_folder,
            is_gt=False,
            only_return_target_ply=False if round_idx == 0 else True,
            reference_renderings=None,
            input_depth_masks=pred_in_depth_masks,
            output_depth_masks=pred_tar_depth_masks,
        )

        vis_all_imgs(
            in_view_rgbs=in_view_rgbs,
            tar_view_rgbs=fake_tar_rgbs,
            in_view_lay_deps=in_lay_depth_images if opt.use_layout_prior else None,
            tar_view_lay_deps=tar_lay_depth_images if opt.use_layout_prior else None,
            in_view_lay_sems=in_lay_sem_images if opt.use_layout_prior else None,
            tar_view_lay_sems=tar_lay_sem_images if opt.use_layout_prior else None,
            in_view_depths=vis_fake_in_depths,
            tar_view_depths=vis_fake_tar_depths,
            in_view_sems=fake_in_sems,
            tar_view_sems=fake_tar_sems,
            output_prefix="all_preds",
            output_folder=output_folder,
        )

        if round_idx == 0:
            room_infer_results["input_poses_metric"] = [mp.cpu().numpy() for mp in in_view_metric_poses]
            room_infer_results["target_poses_metric"] = [mp.cpu().numpy() for mp in tar_view_metric_poses]
            room_infer_results["scene_scale"] = float(batch["scene_scale"][0])

        pts: Float[np.ndarray, "N 3"] = np.asarray(recons_ply.points)
        colors: Float[np.ndarray, "N 3"] = np.asarray(recons_ply.colors)
        semantic_colors: Float[np.ndarray, "N 3"] = np.asarray(recons_sem_ply.colors)
        recons_ply: Float[Tensor, "N 6"] = (
            torch.from_numpy(np.concatenate([pts, colors], axis=-1)).to(dtype=weight_dtype).to(all_rgbs.device)
        )  # N,3
        recons_sem_ply: Float[Tensor, "N 6"] = (
            torch.from_numpy(np.concatenate([pts, semantic_colors], axis=-1)).to(dtype=weight_dtype).to(all_rgbs.device)
        )
        pred_images: Float[Tensor, "bn 3 h w"] = (
            torch.from_numpy(pred_images * 2.0 - 1.0).permute(0, 3, 1, 2).to(dtype=weight_dtype).to(all_rgbs.device)
        )  # B*(n_tasks*T_out + (n_tasks -1)*T_in),3,H,W

    return recons_ply, pred_images.clamp(-1.0, 1.0), room_infer_results, recons_sem_ply


@torch.no_grad()
def inference_controlnet(
    controlnet_pipeline: FluxControlPipeline,
    input_layout_semantics: Float[Tensor, "B N 3 H W"],
    input_layout_depths: Float[Tensor, "B N 3 H W"],
    prompts: List[str],
    num_infer_steps: int = 20,
    generator: torch.Generator = None,
) -> Float[Tensor, "B N 3 H W"]:

    assert (
        input_layout_semantics.shape[1] == input_layout_depths.shape[1] == len(prompts) == 1
    ), "Only support num_view = 1"

    # generate image
    infer_images = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        control_image = (input_layout_semantics[0, i] * 0.5 + 0.5).cpu()
        control_image_pil = torchvision.transforms.ToPILImage()(control_image)
        image = controlnet_pipeline(
            prompt,
            num_inference_steps=num_infer_steps,
            generator=generator,
            control_image=control_image_pil,
            controlnet_conditioning_scale=0.7,
            guidance_scale=3.5,
            height=1024,
            width=1024,
            num_images_per_prompt=1,
        ).images[0]
        print(f"image {i} generated: {image.size}")
        image = image.resize((512, 512), Image.LANCZOS)
        infer_images.append(torchvision.transforms.ToTensor()(image).unsqueeze(0).unsqueeze(0))

    gc.collect()
    torch.cuda.empty_cache()
    infer_images = torch.cat(infer_images, dim=0).to(input_layout_semantics)
    return infer_images * 2.0 - 1.0


def init_controlnet_pipeline(base_model_path: str, controlnet_path: str, device: torch.device):
    pipe = FluxControlPipeline.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(controlnet_path)
    pipe.to(device)
    return pipe


# @torch.no_grad()
def log_val_autoregressive_from_dataloader(
    validation_dataloader,
    pipeline: SpatialGenDiffusionPipeline,
    controlnet_pipeline: FluxControlPipeline,
    args: argparse.Namespace,
    weight_dtype,
    opt: Options,
    split="val",
    num_tasks=2,
    device=torch.device("cuda"),
    exp_dir: str = "",
    styled_prompt_idx: int = 0,
) -> Dict:
    """
    Autoregressive generate dense views using the 8-View RGB-SCM-Sem MVD model.
    1. The dataloader give 16-View tuples;
    2. Generate 8-View tuples using the 8-View RGB-SCM-Sem MVD model iteratively:
        2.1. Generate the first 8-View tuple of RGB-Scm-Sem images: (v0, v1, v2, v3, v4, v5, v6, v7);
        2.2. Get the global SCM from the first 8-Views and mantainance it.
        2.3. Iteratively generate the rest 8-views tuples using the global SCM.
            2.3.1. Select the reference view from the generated views: TODO:(use the global SCM to select the reference view);
            2.3.2. Rendering the layout_semantic and layout_depth images from layout mesh for current 8-View tuple;
            2.3.3. Render the warpped images for target views;
            2.3.4. Generate the current 8-View tuple of RGB-Scm-Sem images;
            2.3.5. Align the generated scm with the global SCM;
            2.3.6. Update the global SCM from the current 8-Views;
            2.3.7. Generate the rest 8 pairs iteratively.
    Args:
        validation_dataloader: the dataloader for validation
        pipeline: the pipeline for inference
        args: the arguments
        weight_dtype: the weight dtype
        opt: the options
        split: the split name (train, val, test)
        num_tasks: number of tasks
        device: the device to use
        exp_dir: the experiment directory
    """
    logger.info("Running {} validation... ".format(split))

    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    output_dir = os.path.join(exp_dir, f"{split}")
    os.makedirs(output_dir, exist_ok=True)

    num_samples = opt.num_views
    total_infer_results = {}

    for valid_step, batch in enumerate(validation_dataloader):
        batch = todevice(batch, device=device)
        room_uid = batch["room_uid"][0].replace("/", "_")
        if TARGET_ROOM_UIDS_PROMPTS is not None:
            if room_uid not in list(TARGET_ROOM_UIDS_PROMPTS.keys()):
                print(f"Skip {room_uid}")
                continue
            else:
                prompts = TARGET_ROOM_UIDS_PROMPTS[room_uid]
                if isinstance(prompts, str):
                    prompts = [prompts]
                elif isinstance(prompts, list):
                    prompts = prompts[styled_prompt_idx : styled_prompt_idx + 1]
                    logger.info(f"styled_prompt: {prompts}")
                else:
                    raise NotImplementedError
                # pass

        infer_results = {}
        logger.info(f"Processing {room_uid} -----------------")

        bsz = batch["image_input"].shape[0]
        T_in = batch["image_input"].shape[1]
        T_out = batch["image_target"].shape[1]
        assert bsz == 1, "Currently only support bsz == 1"
        assert opt.prediction_types == ["rgb", "depth", "semantic"], "Only RGB-D-Semantic task is supported"

        all_rgbs = torch.cat([batch["image_input"][0:1], batch["image_target"][0:1]], dim=1).to(
            dtype=weight_dtype
        )  # B,num_samples,3,H,W
        all_c2w_poses = torch.cat([batch["pose_in"][0:1], batch["pose_out"][0:1]], dim=1)  # B,num_samples,4,4

        ######################################################################
        ### Step 1: 1in7out -- first generate 8 anchor views (RGB, depth and semantic) ###
        ######################################################################
        num_in_views = opt.num_input_views
        num_out_views = 16 - opt.num_input_views
        num_views_per_round = num_in_views + num_out_views
        generated_view_ids = []
        rest_view_ids = []

        num_round = int(1 + np.ceil((num_samples - num_views_per_round) / 1))

        global_scene_ply, global_scene_sem_ply = None, None
        pred_images_lst: List[Float[Tensor, "BN 3 H W"]] = []
        pred_rgbs_list: List[Float[Tensor, "B N 3 H W"]] = []  # generated rgbs
        pred_scms_list: List[Float[Tensor, "B N 3 H W"]] = []  # generated scene_coord_map
        pred_sems_list: List[Float[Tensor, "B N 3 H W"]] = []  # generated semantic maps

        room_uid = batch["room_uid"][0].replace("/", "_")
        room_output_folder = os.path.join(output_dir, room_uid)
        os.makedirs(room_output_folder, exist_ok=True)

        ######################################################################
        ### Step 2: 2in6out -- iteratively generate 8 view tuples, use the generated views as inputs ###
        ######################################################################
        reference_rgb = all_rgbs[0:1, 0:1]
        reference_scm = None
        reference_sem = None

        avg_time = 0
        for round_idx in range(0, num_round):
            begin_tms = time.time()
            # select the refrence view and target view from the 80-View tuples, using viewpoint distance strategy;
            if round_idx == 0:
                # first pair of view ids
                selected_view_ids = [i for i in range(0, num_views_per_round)]
            else:
                # always use the first view as reference
                selected_view_ids = (
                    [0] + generated_view_ids[-(num_in_views - 1) :] + list(rest_view_ids)[:num_out_views]
                )

            round_output_folder = os.path.join(room_output_folder, f"round_{round_idx}")
            os.makedirs(round_output_folder, exist_ok=True)
            in_view_ids = selected_view_ids[:num_in_views]
            tar_view_ids = selected_view_ids[num_in_views:num_views_per_round]
            logger.info(f"Round {round_idx} in_view_ids: {in_view_ids}, tar_view_ids: {tar_view_ids}")

            if len(pred_rgbs_list) == 0:
                in_view_rgbs = all_rgbs[0:1, in_view_ids]
                if args.use_controlnet:
                    all_layout_sems = torch.cat(
                        [batch["semantic_layout_input"][0:1], batch["semantic_layout_target"][0:1]], dim=1
                    ).to(
                        dtype=weight_dtype
                    )  # B,num_samples,3,H,W
                    all_layout_scms = torch.cat(
                        [batch["depth_layout_input"][0:1], batch["depth_layout_target"][0:1]], dim=1
                    ).to(
                        dtype=weight_dtype
                    )  # B,num_samples,3,H,W
                    in_view_scms = all_layout_scms[0:1, in_view_ids]
                    in_view_sems = all_layout_sems[0:1, in_view_ids]
                    if validation_dataloader.dataset.load_dataset == ["hypersim"]:
                        # if hypersim dataset, we use the controlnet output by the SceneCraft
                        all_controlnet_rgbs = torch.cat(
                            [batch["controlnet_image_input"][0:1], batch["controlnet_image_target"][0:1]], dim=1
                        ).to(
                            dtype=weight_dtype
                        )  # B,8,3,H,W
                        in_view_rgbs = all_controlnet_rgbs[0:1, in_view_ids]
                        debug_controlnet_img = torch.cat(
                            [all_rgbs[0, in_view_ids] * 0.5 + 0.5, in_view_sems[0] * 0.5 + 0.5, in_view_rgbs[0] * 0.5 + 0.5]
                        )
                    else:
                        in_view_flux_conditions = batch["controlnet_image_input"][0:1, in_view_ids]
                        in_view_rgbs: Float[Tensor, "B Ni 3 H W"] = inference_controlnet(
                            controlnet_pipeline,
                            input_layout_semantics=in_view_flux_conditions,
                            input_layout_depths=in_view_scms,
                            prompts=prompts,
                            generator=generator,
                        )
                        debug_controlnet_img = torch.cat(
                            [all_rgbs[0, in_view_ids] * 0.5 + 0.5, in_view_flux_conditions[0] * 0.5 + 0.5, in_view_rgbs[0] * 0.5 + 0.5]
                        )
                    torchvision.utils.save_image(
                        debug_controlnet_img, os.path.join(round_output_folder, "debug_controlnet.png")
                    )
                    reference_rgb = in_view_rgbs
                pred_rgbs_list.extend([in_view_rgbs[0:1, i : i + 1] for i in range(in_view_rgbs.shape[1])])
                # in_view scm and semantics are None
                in_view_scms = reference_scm
                in_view_sems = reference_sem
                pred_scms_list.extend([None] * num_in_views)
                pred_sems_list.extend([None] * num_in_views)
            else:
                in_view_rgbs: Float[Tensor, "B No 3 H W"] = torch.cat(
                    [reference_rgb[0:1, 0:1]] + [pred_rgbs_list[i] for i in generated_view_ids[-(num_in_views - 1) :]],
                    dim=1,
                )
                in_view_scms: Float[Tensor, "B No 3 H W"] = torch.cat(
                    [reference_scm[0:1, 0:1]] + [pred_scms_list[i] for i in generated_view_ids[-(num_in_views - 1) :]],
                    dim=1,
                )
                in_view_sems: Float[Tensor, "B No 3 H W"] = torch.cat(
                    [reference_sem[0:1, 0:1]] + [pred_sems_list[i] for i in generated_view_ids[-(num_in_views - 1) :]],
                    dim=1,
                )

            scm_conf_thresh = 6.0 + np.exp(round_idx / 20.0)
            scm_conf_thresh = min(scm_conf_thresh, 8.6)
            round_i_ply, round_i_pred_imgs, room_infer_results, round_i_sem_ply = model_inference(
                args=args,
                round_idx=round_idx,
                opt=opt,
                pipeline=pipeline,
                weight_dtype=weight_dtype,
                in_view_ids=in_view_ids,
                tar_view_ids=tar_view_ids,
                num_tasks=num_tasks,
                generator=generator,
                batch=batch,
                scene_ply=global_scene_ply,
                output_dir=round_output_folder,
                in_view_rgbs=in_view_rgbs,
                in_view_scms=in_view_scms,
                in_view_sems=in_view_sems,
                scm_conf_thresh=scm_conf_thresh,
            )
            if round_idx == 0:
                global_scene_ply = round_i_ply  # N,6
                global_scene_sem_ply = round_i_sem_ply  # N,6
                infer_results[batch["room_uid"][0]] = room_infer_results
            else:
                global_scene_ply = torch.cat((global_scene_ply, round_i_ply), dim=0)
                global_scene_sem_ply = torch.cat((global_scene_sem_ply, round_i_sem_ply), dim=0)
                # append target views infos
                assert (
                    len(infer_results[batch["room_uid"][0]]["target_poses"]) > 0
                ), f"room {room_uid} should get inempty global scene at round_{round_idx}"
                infer_results[batch["room_uid"][0]]["target_rgbs"].extend(room_infer_results["target_rgbs"])
                infer_results[batch["room_uid"][0]]["target_semantics"].extend(room_infer_results["target_semantics"])
                infer_results[batch["room_uid"][0]]["target_depths"].extend(room_infer_results["target_depths"])
                infer_results[batch["room_uid"][0]]["target_poses"].extend(room_infer_results["target_poses"])
                infer_results[batch["room_uid"][0]]["target_points"].extend(room_infer_results["target_points"])
                infer_results[batch["room_uid"][0]]["target_colors"].extend(room_infer_results["target_colors"])

            # put predicted_target_view rgbs to pred_rgbs_lst
            tar_rgb_idx = len(tar_view_ids)
            pred_tar_rgbs = round_i_pred_imgs[:tar_rgb_idx]
            pred_tar_rgbs: Float[Tensor, "No 1 3 H W"] = rearrange(
                pred_tar_rgbs, "(b t) c h w -> t b c h w", b=bsz, t=len(tar_view_ids)
            )
            pred_rgbs_list.extend([pred_tar_rgbs[i].unsqueeze(0) for i in range(pred_tar_rgbs.shape[0])])
            logger.info(f"pred_tar_rgbs shape: {len(pred_tar_rgbs)}")
            # put predicted_target_view depths to pred_depths_lst
            tar_scm_idx = tar_rgb_idx + len(tar_view_ids)
            tar_sem_idx = tar_scm_idx + len(tar_view_ids)
            in_depth_idx = tar_sem_idx + len(in_view_ids)
            in_sem_idx = in_depth_idx + len(in_view_ids)
            pred_tar_scms = round_i_pred_imgs[tar_rgb_idx:tar_scm_idx]
            pred_tar_scms: Float[Tensor, "No 1 3 H W"] = rearrange(
                pred_tar_scms, "(b t) c h w -> t b c h w", b=bsz, t=len(tar_view_ids)
            )
            pred_scms_list.extend([pred_tar_scms[i].unsqueeze(0) for i in range(pred_tar_scms.shape[0])])
            # put predicted_target_view sems to pred_sems_list
            pred_tar_sems = round_i_pred_imgs[tar_scm_idx:tar_sem_idx]
            pred_tar_sems: Float[Tensor, "No 1 3 H W"] = rearrange(
                pred_tar_sems, "(b t) c h w -> t b c h w", b=bsz, t=len(tar_view_ids)
            )
            pred_sems_list.extend([pred_tar_sems[i].unsqueeze(0) for i in range(pred_tar_sems.shape[0])])

            pred_images_lst.append(round_i_pred_imgs)

            # update reference depth and semantic maps
            if reference_scm is None and reference_sem is None:
                pred_in_scms = round_i_pred_imgs[tar_sem_idx:in_depth_idx]
                pred_in_scms: Float[Tensor, "B Ni 3 H W"] = rearrange(
                    pred_in_scms, "(b t) c h w -> b t c h w", b=bsz, t=len(in_view_ids)
                )
                reference_scm = pred_in_scms

                pred_in_sems = round_i_pred_imgs[in_depth_idx:in_sem_idx]
                pred_in_sems: Float[Tensor, "B Ni 3 H W"] = rearrange(
                    pred_in_sems, "(b t) c h w -> b t c h w", b=bsz, t=len(in_view_ids)
                )
                reference_sem = pred_in_sems

            generated_view_ids.extend(selected_view_ids)
            rest_view_ids = np.setdiff1d(np.arange(num_samples), generated_view_ids)
            logger.info(f"rest_view_ids: {rest_view_ids}")

            # TODO: hard code for 4in12out generation, need to be changed
            if len(rest_view_ids) >= 12:
                num_in_views = 4
                num_out_views = 12
            else:
                num_in_views = 16 - len(rest_view_ids)
                num_out_views = len(rest_view_ids)

            end_tms = time.time()
            avg_time += end_tms - begin_tms
            if round_idx == num_round - 1 or len(rest_view_ids) == 0:
                # save the global scene ply
                ply_path = os.path.join(room_output_folder, f"global_scene_ply.ply")
                pcl = o3d.geometry.PointCloud()
                pcl.points = o3d.utility.Vector3dVector(global_scene_ply[:, :3].cpu().numpy())
                pcl.colors = o3d.utility.Vector3dVector(global_scene_ply[:, 3:].cpu().numpy())
                o3d.io.write_point_cloud(ply_path, pcl)

                # save the global scene semantic ply
                ply_path = os.path.join(room_output_folder, f"global_scene_sem_ply.ply")
                pcl = o3d.geometry.PointCloud()
                pcl.points = o3d.utility.Vector3dVector(global_scene_sem_ply[:, :3].cpu().numpy())
                pcl.colors = o3d.utility.Vector3dVector(global_scene_sem_ply[:, 3:].cpu().numpy())
                o3d.io.write_point_cloud(ply_path, pcl)
                torch.cuda.empty_cache()
                gc.collect()
                break

        avg_time /= num_round
        logger.info(f"average inference time: {avg_time}")
        ##################
        ### Step 3: save inference results for gaussian reconstruction and video rendering ###
        ##################
        save_bg_time = time.time()
        filepath = os.path.join(room_output_folder, "inference_results.npz")
        np.savez(filepath, **infer_results)
        logger.info(f"save inference results time: {time.time() - save_bg_time}")
        #  --------------- Convert camera trajectory to metric aligned to the layout ---------------
        infer_results = np.load(filepath, allow_pickle=True)
        for room_uid, room_infer_results in infer_results.items():
            room_infer_results = infer_results[room_uid][()]
            # export camera trajectory in metric
            export_metric_camera_trajectory(
                room_uid=room_uid, room_infer_results=room_infer_results, output_folder=room_output_folder, fps=16
            )

    torch.cuda.empty_cache()
    gc.collect()
    return total_infer_results


def export_metric_camera_trajectory(
    room_uid: str, room_infer_results: Dict[str, Any], output_folder: str, fps: int = 16
):
    """
    Export the GS rendering camera trajectory in metric aligned to the layout.

    Args:
        room_uid: str, the room uid
        room_infer_results: Dict[str, Any], the room infer results
        output_folder: str
        fps: int, interpolation interval between two-frames
    """

    cam_traj: Dict[str, Any] = sparseradegs_interpolate_trajectory(room_infer_results, nframes_interval=fps)
    # export camera trajectory in layout_bbox.ply
    cameras_in_layout = {}
    v0_metric_c2w_pose = room_infer_results["input_poses_metric"][0]
    metric_c2w_poses_in_layout = []
    # we should de-normalized to metric pose
    scale_mat = np.eye(4, dtype=np.float32)
    scale_mat[:3] *= 1.0 / room_infer_results["scene_scale"]
    logger.info(f"Scene scale: {room_infer_results['scene_scale']}")
    import trimesh

    for cam_id, cam_c2w in cam_traj.items():
        cam_c2w = np.array(cam_c2w).reshape(4, 4)
        # scale pose_c2w
        scaled_cam_c2w = scale_mat @ cam_c2w
        R_c2w = scaled_cam_c2w[:3, :3]
        q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
        q_c2w = trimesh.transformations.unit_vector(q_c2w)
        R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
        cam_c2w[:3, :3] = R_c2w
        c2w_pose_in_layout = v0_metric_c2w_pose @ cam_c2w
        # scale  to metric
        metric_c2w_poses_in_layout.append(c2w_pose_in_layout)
        cameras_in_layout[cam_id] = c2w_pose_in_layout.tolist()
    json.dump(cameras_in_layout, open(output_folder + "/cameras_in_layout.json", "w"))
    print(f"Finished exporting camera trajectory for [{room_uid}]......")
    metric_c2w_poses_in_layout = np.array(metric_c2w_poses_in_layout)
    min_position_x = np.min(metric_c2w_poses_in_layout[:, 0, 3])
    min_position_y = np.min(metric_c2w_poses_in_layout[:, 1, 3])
    min_position_z = np.min(metric_c2w_poses_in_layout[:, 2, 3])
    max_position_x = np.max(metric_c2w_poses_in_layout[:, 0, 3])
    max_position_y = np.max(metric_c2w_poses_in_layout[:, 1, 3])
    max_position_z = np.max(metric_c2w_poses_in_layout[:, 2, 3])
    print(
        f"Room bbox: x: [{min_position_x}, {max_position_x}], y: [{min_position_y}, {max_position_y}], z: [{min_position_z}, {max_position_z}]"
    )
    torch.cuda.empty_cache()
    gc.collect()


def sparseradegs_interpolate_trajectory(data_dict: Dict[str, Any], nframes_interval: int = -1) -> List[np.array]:
    input_c2w_poses, target_c2w_poses = data_dict["input_poses"], data_dict["target_poses"]
    all_c2w_poses = np.concatenate([np.array(input_c2w_poses), np.array(target_c2w_poses)], axis=0)
    print(f"Number of views: {all_c2w_poses.shape}")
    num_views = all_c2w_poses.shape[0]

    # interpolate
    if nframes_interval == -1:
        nframes_interval = 24

    interp_radegs_cam_poses = []
    for idx in range(num_views - 1):
        # c2w pose
        c2w_poses = all_c2w_poses[idx : idx + 2]
        interp_c2w_poses = interp_poses(c2w_poses, nframes_interval)
        interp_radegs_cam_poses.extend(interp_c2w_poses)

    interp_radegs_cam_poses_dict = {str(idx): cam_pose for idx, cam_pose in enumerate(interp_radegs_cam_poses)}
    return interp_radegs_cam_poses_dict


def main():
    import multiprocessing

    multiprocessing.set_start_method("spawn")  # 必须在所有多进程操作前调用

    parser = argparse.ArgumentParser(description="Infer a diffusion model for 3D Scene generation")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    parser.add_argument("--tag", type=str, default=None, help="Tag that refers to the current experiment")
    parser.add_argument("--output_dir", type=str, default="out", help="Path to the output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the PRNG")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG scale used for validation")
    parser.add_argument("--half_precision", action="store_true", help="Use half precision for inference")
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 for faster training on Ampere GPUs")

    parser.add_argument("--image_path", type=str, default=None, help="Path to the image for reconstruction")
    parser.add_argument(
        "--image_dir", type=str, default=None, help="Path to the directory of images for reconstruction"
    )
    parser.add_argument("--infer_from_iter", type=int, default=-1, help="The iteration to load the checkpoint from")
    parser.add_argument("--scheduler_type", type=str, default="sde-dpmsolver++", help="Type of diffusion scheduler")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Diffusion steps for inference")
    parser.add_argument(
        "--triangle_cfg_scaling",
        action="store_true",
        help="Whether or not to use triangle classifier-free guidance scaling",
    )
    parser.add_argument("--min_guidance_scale", type=float, default=1.0, help="Minimum of triangle cfg scaling")
    parser.add_argument("--eta", type=float, default=1.0, help="The weight of noise for added noise in diffusion step")
    parser.add_argument(
        "--infer_tag",
        type=str,
        default="35k_inference_16view",
        help="The inference tag",
    )
    parser.add_argument(
        "--use_controlnet",
        action="store_true",
        help="Whether or not to use controlnet for inference",
    )
    parser.add_argument(
        "--styled_prompt_idx",
        type=int,
        default=0,
        help="The index of styled prompt for inference",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="scene_00000",
        help="selected scene id to generate",
    )
    parser.add_argument(
        "--style_prompt",
        type=str,
        default="A Traditional Chinese Style living room with rosewood furniture, jade ornaments, and silk screens, \
            arranged in a feng shui layout with a central rosewood coffee table and a black lacquer sideboard. \
            Warm, natural lighting enhances the deep red and gold accents, while paper lanterns and carved details add to the aesthetic. \
            The color palette includes deep red, gold, black lacquer, jade green, and warm brown, creating a harmonious and elegant atmosphere.",
        help="text prompt for FLUX-Controlnet"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    print(f"Options: {opt}")

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = (
            time.strftime("%Y-%m-%d_%H:%M") + "_" + os.path.split(args.config_file)[-1].split()[0]
        )  # config file name

    # Create the experiment directory
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    infer_dir = os.path.join(exp_dir, args.infer_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(infer_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO)

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    ray_encoder = RayMapEncoder.from_pretrained_new(opt.pretrained_model_name_or_path, subfolder="ray_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
    scm_vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scm_vae")

    num_tasks = len(opt.prediction_types) + 2 if opt.use_layout_prior else len(opt.prediction_types)
    # Initialize the model
    in_channels = 4  # hard-coded for SD 1.5/2.1
    if opt.input_concat_plucker:
        in_channels += 16
    if opt.input_concat_binary_mask:
        in_channels += 1
    if opt.input_concat_warpped_image:
        in_channels += 4
    logger.info(
        f"input_concat_plucker: {opt.input_concat_plucker}, input_concat_binary_mask: {opt.input_concat_binary_mask}, input_concat_warpped_image: {opt.input_concat_warpped_image}"
    )
    unet_from_pretrained_kwargs = {
        "sample_size": opt.input_res // 8,  # `8` hard-coded for SD 1.5/2.1
        "in_channels": in_channels,
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
        "input_concat_warpped_image": opt.input_concat_warpped_image,
        "num_input_views": opt.num_input_views,
        "num_output_views": 16 - opt.num_input_views,
        "num_tasks": num_tasks,
        "cd_attention_mid": num_tasks > 1 and opt.enable_mm_attn,
        "multiview_attention": True,
        "sparse_mv_attention": False,
        "disable_mv_attention_in_64x64": opt.input_res == 512,
    }
    unet, loading_info = UNetMVMM2DConditionModel.from_pretrained_new(
        opt.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        **unet_from_pretrained_kwargs,
    )

    if args.scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    elif "dpmsolver" in args.scheduler_type:
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler.config.algorithm_type = args.scheduler_type
    elif args.scheduler_type == "edm":
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            opt.pretrained_model_name_or_path, subfolder="scheduler"
        )
    else:
        raise NotImplementedError(f"Scheduler [{args.scheduler_type}] is not supported by now")
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    # Freeze all models
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    scm_vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    scm_vae.eval()
    unet.eval()

    vae.to(device, dtype=weight_dtype)
    scm_vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    pipeline = SpatialGenDiffusionPipeline(
        vae=vae,
        depth_vae=scm_vae,
        unet=unet,
        ray_encoder=ray_encoder,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline.unet.enable_xformers_memory_efficient_attention()

    pipeline.to(device).to(dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    if args.use_controlnet:
        controlnet_pipeline = init_controlnet_pipeline(
            base_model_path="black-forest-labs/FLUX.1-dev",
            controlnet_path="manycore-research/FLUX.1-Wireframe-dev-lora",
            device=device,
        )
    else:
        controlnet_pipeline = None

    val_dataset = ExampleDataset(
        data_dir=opt.test_data_dir,
        dataset_name=opt.dataset_names[0],
        split_filepath=opt.test_split_file,
        image_height=opt.input_res,
        image_width=opt.input_res,
        T_in=opt.num_input_views,
        total_view=opt.num_views,
        validation=True,
        use_normal="normal" in opt.prediction_types,
        use_semantic="semantic" in opt.prediction_types or opt.use_layout_prior,
        use_metric_depth=opt.use_metric_depth,
        use_scene_coord_map=opt.use_scene_coord_map,
        use_layout_prior=opt.use_layout_prior,
        use_layout_prior_from_p3d=opt.use_layout_prior,
        return_metric_data=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    logger.info(f"Load [{len(val_loader)}] validation samples\n")

    global TARGET_ROOM_UIDS_PROMPTS
    TARGET_ROOM_UIDS_PROMPTS = None
    if val_dataset.load_dataset == "spatialgen":
        TARGET_ROOM_UIDS_PROMPTS = {args.scene_id: args.style_prompt}
        logger.info(f"TARGET_ROOM_UIDS_PROMPTS: {TARGET_ROOM_UIDS_PROMPTS}")
    elif val_dataset.load_dataset == "spatiallm":
        raise NotImplementedError("spatiallm dataset is not supported yet.")

    rooms_infer_res_dict = log_val_autoregressive_from_dataloader(
        validation_dataloader=val_loader,
        pipeline=pipeline,
        controlnet_pipeline=controlnet_pipeline,
        args=args,
        opt=opt,
        split="val",
        num_tasks=num_tasks,
        device=device,
        weight_dtype=weight_dtype,
        exp_dir=infer_dir,
        styled_prompt_idx=args.styled_prompt_idx,
    )

    ##################
    ### gaussian reconstruction and video rendering ###
    ##################
    infer_dir = os.path.abspath(infer_dir)
    room_output_folders = [
        f
        for f in os.listdir(infer_dir + "/val")
        if os.path.isdir(os.path.join(infer_dir, "val", f))
        and os.path.exists(os.path.join(infer_dir, "val", f, "inference_results.npz"))
    ]
    os.chdir("./src/recons/Sparse-RaDeGS")
    for room_output_folder in room_output_folders:
        logger.info(f"Reconstructing scene: {room_output_folder}")
        gs_train_cmd_str = (
            "python3 train.py "
            + f"--source_path {infer_dir}/val/{room_output_folder} "
            + f"--model_path {infer_dir}/val/{room_output_folder}/sparseradegs_out "
            + "--beta 5.0 "
            + "--lambda_warp_reg 0.4 "
            + "--iterations 7000 "
            + "-r 1 "
            + "--warp_reg_start_itr 3000 "
        )
        print(gs_train_cmd_str)
        os.system(gs_train_cmd_str)

        gs_render_cmd_str = (
            "python3 render.py "
            + f"--source_path {infer_dir}/val/{room_output_folder} "
            + f"--model_path {infer_dir}/val/{room_output_folder}/sparseradegs_out "
            + "--no_load_depth "
            + "--iteration -1 "
        )
        print(gs_render_cmd_str)
        os.system(gs_render_cmd_str)
        
if __name__ == "__main__":
    main()
