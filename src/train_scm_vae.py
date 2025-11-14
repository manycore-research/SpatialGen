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
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.utils import is_xformers_available
from transformers import AutoConfig
from torchvision.utils import save_image

from src.options import opt_dict, Options
from src.data import MixDataset
from src.models import (
    get_optimizer,
    get_lr_scheduler,
)

import src.utils.util as util
from src.utils.typing import *
from src.utils.pcl_ops import save_input_output_pointcloud
from src.utils.vis_util import apply_depth_to_colormap, save_color_depth_image
from src.utils.misc import print_memory, worker_init_fn, enable_flash_attn_if_avail

from diffusers_spatialgen import MyEMAModel

os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_autotune"

logger = get_accelerate_logger(__name__, log_level="INFO")

TAE_DICT = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "madebyollin/taesd",
    "stabilityai/stable-diffusion-2-1": "madebyollin/taesd",
    "PixArt-alpha/PixArt-XL-2-512x512": "madebyollin/taesd",
    "stabilityai/stable-diffusion-xl-base-1.0": "madebyollin/taesdxl",
    "madebyollin/sdxl-vae-fp16-fix": "madebyollin/taesdxl",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "madebyollin/taesdxl",
    "stabilityai/stable-diffusion-3-medium-diffusers": "madebyollin/taesd3",
    "stabilityai/stable-diffusion-3.5-medium": "madebyollin/taesd3",
    "stabilityai/stable-diffusion-3.5-large": "madebyollin/taesd3",
    "black-forest-labs/FLUX.1-dev": "madebyollin/taef1",
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers": "madebyollin/taesdxl",
}


def expand_vae_output(ori_vae, is_tiny=False):
    """ Expand the output channels of the VAE to 4 channels. The original VAE output is 3 channels.

    Args:
        pretrained_model_name_or_path (str): repoid
        filename (_type_): path to the config file
        ori_vae (_type_): original VAE model

    Returns:
        vae: _description_
    """
    vae_config = {**ori_vae.config}
    vae_config["out_channels"] = 4
    if not is_tiny:
        m = AutoencoderKL.from_config(vae_config)
    else:
        m = AutoencoderTiny.from_config(vae_config)
    for k,v in ori_vae.state_dict().items():
        if v.shape != m.state_dict()[k].shape:
            print(f"[expand_vae_output] {k} shape mismatch: {v.shape} vs {m.state_dict()[k].shape}")
            # zero init
            torch.nn.init.zeros_(m.state_dict()[k])
            m.state_dict()[k][:3,...].copy_(v)
        elif k in m.state_dict():
            m.state_dict()[k].copy_(v)
        else:
            print(f"not found {k}")
    return m

# ref VGGT: https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/training/loss.py#L370
def scm_gradient_loss(prediction, target, conf=None, gamma=1.0, alpha=0.2):
    # prediction: B, H, W, C
    # target: B, H, W, C

    mask = torch.ones(prediction.shape[:-1], dtype=torch.float32, device=prediction.device)[:, :, :, None]
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        gamma = 1.0
        alpha = 0.2

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss

def scm_gradient_loss_multi_scale(prediction, target, scales=4, gradient_loss_fn=scm_gradient_loss, conf=None):
    """
    Compute gradient loss across multiple scales
    Args:
        prediction: B, H, W, C
        target: B, H, W, C
        scales: number of scales to compute gradient loss
        gradient_loss_fn: function to compute gradient loss
        conf: confidence map
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            # mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )

    total = total / scales
    return total


@torch.no_grad()
def log_validation(
    validation_dataloader,
    vae: Union[AutoencoderTiny, AutoencoderKL],
    args,
    opt: Options,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    split: str = "val",
    exp_dir: str = "./",
):
    logger.info("Running {} validation... ".format(split))

    image_logs = []

    output_dir = os.path.join(exp_dir, f"{split}")
    os.makedirs(output_dir, exist_ok=True)

    def convert_depth_to_colormap(depths: Float[Tensor, "BNt C H W"]):
        depths = depths.permute(0, 2, 3, 1).float().cpu()
        depths = ((depths + 1.0) / 2.0).clip(0, 1)
        depths = depths.mean(dim=-1)
        color_depths = apply_depth_to_colormap(depths, cmap="viridis")
        return torch.from_numpy(color_depths).permute(0, 3, 1, 2) * 2.0 - 1.0

    # original VAE mode
    ori_vae: AutoencoderKL = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
    ori_vae.eval()
    ori_vae.requires_grad_(False)

    ori_vae.to(accelerator.device, dtype=weight_dtype)

    for valid_step, batch in enumerate(validation_dataloader):
        if args.max_val_steps is not None and valid_step >= args.max_val_steps:
            break

        T_in = batch["image_input"].shape[1]
        T_out = batch["image_target"].shape[1]
        input_rgbs = batch["image_input"][0:1].to(dtype=weight_dtype)  # B,T,3,H,W
        input_depths = batch["depth_input"][0:1].to(dtype=weight_dtype)  # B,T_in,3,H,W

        gt_images = torch.cat([input_rgbs, input_depths], dim=1)  # B,2T,3,H,W
        gt_images = rearrange(gt_images, "b t c h w -> (b t) c h w")
        
        room_uid = batch["room_uid"][0].replace("/", "_")
        gt_output_dir = os.path.join(output_dir, f"{room_uid}")
        ori_vae_output_dir = os.path.join(output_dir, f"{room_uid}_ori")
        ft_vae_output_dir = os.path.join(output_dir, f"{room_uid}_ft")
        os.makedirs(gt_output_dir, exist_ok=True)
        os.makedirs(ori_vae_output_dir, exist_ok=True)
        os.makedirs(ft_vae_output_dir, exist_ok=True)
        with torch.autocast("cuda"):
            rgb_idx, depth_idx = 0, T_in

            latents = (ori_vae.encode(gt_images).latent_dist.mode() * ori_vae.config.scaling_factor).to(weight_dtype)
            latents = latents if args.use_tiny_vae else latents / ori_vae.config.scaling_factor
            reconstructions_ft = vae.decode(latents).sample.to(weight_dtype)
            reconstructions_conf = reconstructions_ft[:, 3:, :, :]
            logger.info(f"reconstructions shape: {reconstructions_ft.shape}, reconstructions_conf shape: {reconstructions_conf.shape}")
            reconstructions_ft = reconstructions_ft[:, :3, :, :]
            reconstructions = ori_vae.decode(latents / ori_vae.config.scaling_factor).sample.to(weight_dtype)

            recons_ft_cpu = reconstructions_ft.cpu()
            recons_cpu = reconstructions.cpu()
            gts_cpu = gt_images.cpu()

            ft_fake_depths = recons_ft_cpu[depth_idx:, :, :, :].clone()
            if args.predict_conf:
                ft_fake_depth_confs = reconstructions_conf[depth_idx:, :, :, :].cpu().clone()
                logger.info(f"ft_fake_depth_confs min: {ft_fake_depth_confs.min()}, median: {ft_fake_depth_confs.median()}, mean: {ft_fake_depth_confs.mean()} max: {ft_fake_depth_confs.max()}")
                logger.info(f"exp ft_fake_depth_confs min: {(1+ft_fake_depth_confs.exp()).min()}, median: {(1+ft_fake_depth_confs.exp()).median()}, max: {(1+ft_fake_depth_confs.exp()).max()}")
                torchvision.utils.save_image(ft_fake_depth_confs, f"{ft_vae_output_dir}/conf.png", normalize=True)
            ori_fake_depths = recons_cpu[depth_idx:, :, :, :].clone()
            gt_depths = gts_cpu[depth_idx:, :, :, :].clone()

            # compute point map gradient loss
            gradient_loss = scm_gradient_loss_multi_scale(prediction=ft_fake_depths.permute(0, 2, 3, 1), 
                                                          target=gt_depths.permute(0, 2, 3, 1), 
                                                          scales=4)
            logger.info(f"Gradient loss: {gradient_loss}")

            # convert depth into colorful depth
            if not opt.use_scene_coord_map:
                recons_ft_cpu[depth_idx:, :, :, :] = convert_depth_to_colormap(recons_ft_cpu[depth_idx:, :, :, :])
                recons_cpu[depth_idx:, :, :, :] = convert_depth_to_colormap(recons_cpu[depth_idx:, :, :, :])
                gts_cpu[depth_idx:, :, :, :] = convert_depth_to_colormap(gts_cpu[depth_idx:, :, :, :])

            pose_in = batch["pose_in"][0:1]  # B,T_in,4,4
            pose_out = batch["pose_out"][0:1]  # B,T_out,4,4
            min_depth = batch["depth_min"][0:1].cpu()  # B,1
            max_depth = batch["depth_max"][0:1].cpu()
            scene_scale = batch["scene_scale"][0:1].cpu()
            intrinsic_mat = batch["intrinsic"][0].cpu().numpy()
            logger.info("validating RGB-D task on room {}".format(room_uid))

            in_rgbs = rearrange(input_rgbs.float(), "b t c h w -> (b t) c h w")
            gt_in_depths = gt_depths[:, 0:1, :, :].float() if not opt.use_scene_coord_map else gt_depths.float()
            ori_fake_in_depths = (
                ori_fake_depths[:, 0:1, :, :].float() if not opt.use_scene_coord_map else ori_fake_depths.float()
            )
            ft_fake_in_depths = (
                ft_fake_depths[:, 0:1, :, :].float() if not opt.use_scene_coord_map else ft_fake_depths.float()
            )

            save_input_output_pointcloud(
                input_images=in_rgbs,
                input_depths=gt_in_depths,
                output_images=in_rgbs,
                output_depths=gt_in_depths,
                poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=T_in),
                poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=T_out),
                min_depth=min_depth.float(),
                max_depth=max_depth.float(),
                scene_scale=scene_scale.float(),
                use_metric_depth=opt.use_metric_depth,
                use_scene_coord_map=opt.use_scene_coord_map,
                output_folder=gt_output_dir,
                intrinsic_mat=intrinsic_mat,
                is_gt=True,
            )

            save_input_output_pointcloud(
                input_images=in_rgbs,
                input_depths=ori_fake_in_depths,
                output_images=in_rgbs,
                output_depths=ori_fake_in_depths,
                poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=T_in),
                poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=T_out),
                min_depth=min_depth.float(),
                max_depth=max_depth.float(),
                scene_scale=scene_scale.float(),
                use_metric_depth=opt.use_metric_depth,
                use_scene_coord_map=opt.use_scene_coord_map,
                output_folder=ori_vae_output_dir,
                intrinsic_mat=intrinsic_mat,
                is_gt=False,
            )
            save_input_output_pointcloud(
                input_images=in_rgbs,
                input_depths=ft_fake_in_depths,
                output_images=in_rgbs,
                output_depths=ft_fake_in_depths,
                poses_input=rearrange(pose_in.float(), "b t c d-> (b t) c d", t=T_in),
                poses_output=rearrange(pose_out.float(), "b t c d -> (b t) c d", t=T_out),
                min_depth=min_depth.float(),
                max_depth=max_depth.float(),
                scene_scale=scene_scale.float(),
                use_metric_depth=opt.use_metric_depth,
                use_scene_coord_map=opt.use_scene_coord_map,
                output_folder=ft_vae_output_dir,
                intrinsic_mat=intrinsic_mat,
                input_depth_conf_maps=ft_fake_depth_confs,
                output_depth_conf_maps=ft_fake_depth_confs,
                is_gt=False,
            )
            # calc metrics
            image_logs.append(
                {
                    "ori_pred_images": (recons_cpu + 1.0) / 2.0,
                    "ft_pred_images": (recons_ft_cpu + 1.0) / 2.0,
                    "gt_images": (gts_cpu + 1.0) / 2.0,
                }
            )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log_id, log in enumerate(image_logs):
                ori_pred_images = log["ori_pred_images"]
                ft_pred_images = log["ft_pred_images"]
                gt_images = log["gt_images"]
                # save to tensorboard
                gt_images = torchvision.utils.make_grid(gt_images, nrow=T_in * 2)
                ft_pred_images = torchvision.utils.make_grid(ft_pred_images, nrow=T_in * 2)
                ori_pred_images = torchvision.utils.make_grid(ori_pred_images, nrow=T_in * 2)

                gt_imgs = gt_images.float().detach().cpu()
                tracker.writer.add_image(
                    f"{split}/{log_id}/gts",
                    gt_imgs,
                    global_step=valid_step,
                )
                pred_imgs = ft_pred_images.float().detach().cpu()
                tracker.writer.add_image(
                    f"{split}/{log_id}/ft_preds",
                    pred_imgs,
                    global_step=valid_step,
                )

                ori_pred_imgs = ori_pred_images.float().detach().cpu()
                tracker.writer.add_image(
                    f"{split}/{log_id}/ori_preds",
                    ori_pred_imgs,
                    global_step=valid_step,
                )

    # after validation, set the pipeline back to training mode
    vae.train()
    return image_logs


def main():
    PROJECT_NAME = "Finetuning SCM-VAE"

    parser = argparse.ArgumentParser(
        description="Train a VariationalAutoEncoder model for Scene Coordinate Map",
    )

    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    parser.add_argument("--tag", type=str, required=True, help="Tag that refers to the current experiment")
    parser.add_argument("--output_dir", type=str, default="out", help="Path to the output directory")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",  # log_image currently only for wandb
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--resume_from_iter", type=int, default=-1, help="The iteration to load the checkpoint from")
    parser.add_argument("--seed", type=int, default=2025, help="Seed for the PRNG")
    parser.add_argument("--max_train_steps", type=int, default=None, help="The max iteration step for training")
    parser.add_argument("--max_val_steps", type=int, default=1, help="The max iteration step for validation")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="The number of processed spawned by the batch provider"
    )
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for the data loader")

    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for training")
    parser.add_argument("--scale_lr", action="store_true", help="Scale lr with total batch size (base batch size: 256)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training",
    )
    parser.add_argument("--allow_tf32", action="store_true", help="Enable TF32 for faster training on Ampere GPUs")

    parser.add_argument(
        "--val_guidance_scales", type=list, nargs="+", default=[2.0], help="CFG scale used for validation"
    )

    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed for training")
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed",
    )
    parser.add_argument(
        "--load_pretrained_gsvae_model",
        type=str,
        default="/project/lrmcongen/codes/fangchuan/DiffSplat/out/gsvae_sd_spiral",
        help="Tag of a pretrained GSVAE in this project",
    )
    parser.add_argument(
        "--use_tiny_vae",
        action="store_true",
        help="Whether to use tiny VAE for training.",
    )
    parser.add_argument(
        "--predict_conf",
        action="store_true",
        help="Whether to predict confidence map for training.",
    )
    parser.add_argument(
        "--conf_gamma",
        type=float,
        default=1.0,
        help="The gamma value for the confidence map loss.",
    )
    parser.add_argument(
        "--conf_alpha",
        type=float,
        default=0.2,
        help="The alpha value for the confidence map loss.",
    )
    parser.add_argument(
        "--load_from_pretrained_ckpt_dir",
        type=str,
        default=None,
        help="Load pretrained checkpoint for training.",
    )
    parser.add_argument(
        "--use_grad_loss",
        action="store_true",
        help="Whether to use gradient loss for training.",
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
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S", level=logging.INFO)

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
        use_normal="normal" in opt.prediction_types,
        use_semantic="semantic" in opt.prediction_types or opt.use_layout_prior,
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
        use_normal="normal" in opt.prediction_types,
        use_semantic="semantic" in opt.prediction_types or opt.use_layout_prior,
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
    total_batch_size = (
        configs["train"]["batch_size_per_gpu"] * accelerator.num_processes * args.gradient_accumulation_steps
    )
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= total_batch_size / 256
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # LPIPS loss
    if accelerator.is_main_process:
        _ = LPIPS(net="vgg")
        del _
    accelerator.wait_for_everyone()  # wait for pretrained backbone weights to be downloaded
    lpips_loss_fn = LPIPS(net="vgg").to(accelerator.device)
    lpips_loss_fn = lpips_loss_fn.requires_grad_(False)
    lpips_loss_fn.eval()
    
    if args.load_from_pretrained_ckpt_dir is None:
        # load pretrained VAE
        sd_vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
        if args.use_tiny_vae:
            tiny_vae = AutoencoderTiny.from_pretrained(TAE_DICT[opt.pretrained_model_name_or_path])
    else:
        if args.use_tiny_vae:
            sd_vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
            tiny_vae = AutoencoderTiny.from_pretrained(args.load_from_pretrained_ckpt_dir)
        else:
            sd_vae = AutoencoderKL.from_pretrained(args.load_from_pretrained_ckpt_dir)
        logger.info(f"Loading pretrained VAE from {args.load_from_pretrained_ckpt_dir}")

    if args.predict_conf:
        sd_vae = expand_vae_output( sd_vae)
        if args.use_tiny_vae:
            tiny_vae = expand_vae_output(tiny_vae, is_tiny=True)
    
    if args.use_ema:
        if args.use_tiny_vae:
            ema_vae = MyEMAModel(
                tiny_vae.parameters(),
                model_cls=AutoencoderTiny,
                model_config=tiny_vae.config,
                **configs["train"]["ema_kwargs"],
            )
        else:
            ema_vae = MyEMAModel(
                sd_vae.parameters(),
                model_cls=AutoencoderKL,
                model_config=sd_vae.config,
                **configs["train"]["ema_kwargs"],
            )
    # only finetune the decoder
    if args.use_tiny_vae:
        sd_vae.encoder.eval()
        sd_vae.encoder.requires_grad_(False)
        sd_vae.decoder.eval()
        sd_vae.decoder.requires_grad_(False)
        tiny_vae.encoder.eval()
        tiny_vae.encoder.requires_grad_(False)
        tiny_vae.decoder.train()
        tiny_vae.decoder.requires_grad_(True)
    else:
        sd_vae.encoder.eval()
        sd_vae.encoder.requires_grad_(False)
        sd_vae.decoder.train()
        sd_vae.decoder.requires_grad_(True)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_vae.save_pretrained(os.path.join(output_dir, "ema_vae"))

        accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

    opt_vae = tiny_vae if args.use_tiny_vae else sd_vae
    # enable efficient training
    enable_flash_attn_if_avail(opt_vae)
    if opt.grad_checkpoint:
            opt_vae.enable_gradient_checkpointing()
    opt_vae.enable_slicing()

    optimizer = get_optimizer(
        params=[
            {"params": opt_vae.parameters(), "lr": configs["optimizer"]["lr"]},
        ],
        **configs["optimizer"],
    )

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps
    )  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu

    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # Prepare everything with `accelerator`
    opt_vae, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        opt_vae, optimizer, lr_scheduler, train_loader, val_loader
    )

    if args.use_ema:
        ema_vae.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # # Move `vae` to gpu and cast to `weight_dtype`
    sd_vae.to(accelerator.device, dtype=weight_dtype)

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
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

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
            accelerator.load_state(
                os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}")
            )  # torch < 2.4.0 here for `weights_only=False`
            if args.use_ema:
                ema_vae.load_state_dict(
                    torch.load(
                        os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}", "ema_vae.pth"),
                        map_location=accelerator.device,
                    )
                )
            global_update_step = int(args.resume_from_iter)
            first_epoch = global_update_step // updated_steps_per_epoch

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)
        util.save_model_architecture(accelerator.unwrap_model(opt_vae), exp_dir)

    # Start training
    progress_bar = tqdm(
        range(total_updated_steps), initial=global_update_step, desc="Training", disable=not accelerator.is_main_process
    )
    for epoch in range(first_epoch, num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(opt_vae):

                opt_vae: Union[AutoencoderTiny, AutoencoderKL]

                input_rgbs = batch["image_input"].to(dtype=weight_dtype)  # B,T,3,H,W
                input_depths = batch["depth_input"].to(dtype=weight_dtype)  # B,T,3,H,W
                target_depths = batch["depth_target"].to(dtype=weight_dtype)  # B,T,3,H,W
                targets = torch.cat([input_depths, target_depths], dim=1)  # B,T_in+T_out 3,H,W
                targets = rearrange(targets, "b t c h w -> (b t) c h w")

                latents = (sd_vae.encode(targets).latent_dist.mode() * sd_vae.config.scaling_factor).to(weight_dtype)
                latents = latents if args.use_tiny_vae else latents / sd_vae.config.scaling_factor
                preds = opt_vae.module.decode(latents).sample.to(weight_dtype)
                preds = opt_vae.module.decode(latents).sample.to(weight_dtype) # (B T) 4 H W

                if args.predict_conf:
                    preds_conf = 1 + preds[:, 3:4, :, :].exp()  # B,T_in+T_out,1,H,W
                    preds = preds[:, 0:3, :, :]  # B,T_in+T_out,3,H,W

                # default loss
                # if not args.predict_conf:
                mse_loss = tF.mse_loss(preds.float(), targets.float(), reduction="none")
                if args.predict_conf:
                    mse_loss = args.conf_gamma * mse_loss * preds_conf - args.conf_alpha * torch.log(preds_conf)
                mse_loss = mse_loss.mean()
                if lpips_loss_fn is not None:
                    lpips_loss = lpips_loss_fn(preds.float(), targets.float()).mean()
                    if not torch.isfinite(lpips_loss):
                        lpips_loss = torch.zeros_like(mse_loss, device=mse_loss.device)
                if args.use_grad_loss:
                    grad_loss = scm_gradient_loss_multi_scale(prediction=preds.permute(0, 2, 3, 1), 
                                                  target=targets.permute(0, 2, 3, 1))
                else:
                    grad_loss = torch.zeros_like(mse_loss, device=mse_loss.device)

                # remove kl term from loss, bc when we only train the decoder, the latent is untouched
                # and the kl loss describes the distribution of the latent
                loss = mse_loss + opt.lpips_weight * lpips_loss + 10.0 * grad_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(opt_vae.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_vae.step(opt_vae.parameters())
                progress_bar.update(1)
                global_update_step += 1

                # Save checkpoint
                if (
                    global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                    or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                    or global_update_step == total_updated_steps
                ):  # 3. last step of an epoch

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

                                logger.info(
                                    f"{len(pipelines)} pipelines already exist, removing {len(removing_pipelines)} pipelines"
                                )
                                logger.info(f"removing pipelines: {', '.join(removing_pipelines)}")

                                for removing_pipeline in removing_pipelines:
                                    removing_pipeline = os.path.join(pipeline_dir, removing_pipeline)
                                    shutil.rmtree(removing_pipeline)

                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_vae.store(opt_vae.parameters())
                            ema_vae.copy_to(opt_vae.parameters())
                        unwrapped_vae = accelerator.unwrap_model(opt_vae)
                        pipeline_save_path = os.path.join(pipeline_dir, f"pipeline-{global_update_step:06d}")
                        unwrapped_vae.save_pretrained(pipeline_save_path)
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_vae.restore(opt_vae.parameters())

                    gc.collect()

                # Evaluate on the validation set
                if accelerator.is_main_process:
                    if (
                        global_update_step == 1
                        or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                        or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                        or global_update_step == total_updated_steps
                    ):  # 4. last step of an epoch

                        torch.cuda.empty_cache()
                        gc.collect()

                        # Use EMA parameters for evaluation
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference
                            ema_vae.store(opt_vae.parameters())
                            ema_vae.copy_to(opt_vae.parameters())
                        train_image_logs = log_validation(
                            train_loader,
                            vae=accelerator.unwrap_model(opt_vae),
                            args=args,
                            opt=opt,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            split="train",
                            exp_dir=exp_dir,
                        )

                        val_image_logs = log_validation(
                            val_loader,
                            vae=accelerator.unwrap_model(opt_vae),
                            args=args,
                            opt=opt,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            split="val",
                            exp_dir=exp_dir,
                        )

                        if args.use_ema:
                            # Switch back to the original UNet parameters
                            ema_vae.restore(opt_vae.parameters())

                        torch.cuda.empty_cache()
                        gc.collect()

            # Log the training progress
            if (
                global_update_step % configs["train"]["log_freq"] == 0
                or global_update_step == 1
                or global_update_step % updated_steps_per_epoch == 0
            ):  # last step of an epoch
                log_loss = loss.detach().item()
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