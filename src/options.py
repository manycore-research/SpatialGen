from typing import *

from dataclasses import dataclass, field
from copy import deepcopy



@dataclass
class Options:
    # Dataset
    input_res: int = 256
        ## Camera
    num_input_views: int = 8
    num_views: int = 16
    trajectory_sampler_type: Literal[
        "line",
        "spiral",
        "panoramic",
        "randomwalk",
    ] = "randomwalk"

    
    # training datasets
    spatialgen_data_dir: str = "/project/lrmcongen/data/roomverse_data/processed_data"
    train_split_file: str = "/project/lrmcongen/data/roomverse_data/new_perspective_trains.txt"
    invalid_split_file: str = "/project/lrmcongen/data/roomverse_data/new_perspective_invalid_scenes.txt"
    prompt_embed_dir: Optional[str] = None  # precompute T5 embedding
    prediction_types: list[str] = field(default_factory=lambda: ["rgb", "depth"])
    use_metric_depth: bool = False
    use_layout_prior: bool = False
    use_scene_coord_map: bool = False
    
    hypersim_data_dir: str = "/project/lrmcongen/data/hypersim/evermotion_dataset/semantic_images2"

    # testing datasets
    dataset_names: List[str] = field(default_factory=lambda: ["spatialgen"])
    test_data_dir: str = "/project/lrmcongen/data/roomverse_data/processed_data"
    test_split_file: str = "/project/lrmcongen/data/roomverse_data/new_perspective_test.txt"
    spatiallm_data_dir: str = "./demo/spatiallm_testset/"
    structured3d_data_dir: str = "./demo/std3d_testset/"


    
    ## Transformer
    llama_style: bool = True
    patch_size: int = 8
    dim: int = 512
    num_blocks: int = 12
    num_heads: int = 8
    grad_checkpoint: bool = True


    # MVD
    pretrained_model_name_or_path: Literal[
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
        "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-medium",
        "stabilityai/stable-diffusion-3.5-large",
        "black-forest-labs/FLUX.1-dev",
        "madebyollin/sdxl-vae-fp16-fix",
        "lambdalabs/sd-image-variations-diffusers",
        "stabilityai/stable-diffusion-2-1-unclip",
        "chenguolin/sv3d-diffusers",
    ] = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    load_fp16vae_for_sdxl: bool = True
        ## Config
    from_scratch: bool = False
    cfg_dropout_prob: float = 0.05  # actual prob is x2; see the training code
    snr_gamma: float = 0.  # Min-SNR trick; `0.` menas not used
    num_inference_steps: int = 20
    noise_scheduler_type: Literal[
        "ddim",
        "dpmsolver++",
        "sde-dpmsolver++",
    ] = "dpmsolver++"
    prediction_type: Optional[str] = None  # `None` means using default prediction type
    beta_schedule: Optional[str] = None  # `None` means using the default beta schedule
    edm_style_training: bool = False  # EDM scheduling; cf. https://arxiv.org/pdf/2206.00364
    common_tricks: bool = True  # cf. https://arxiv.org/pdf/2305.08891 (including: 1. trailing timestep spacing, 2. rescaling to zero snr)
            ### SD3; cf. https://arxiv.org/pdf/2403.03206
    weighting_scheme: Literal[
        "sigma_sqrt",
        "logit_normal",
        "mode",
        "cosmap",
    ] = "logit_normal"
    logit_mean: float = 0.
    logit_std: float = 1.
    mode_scale: float = 1.29
    precondition_outputs: bool = False  # whether prediction x_0
    ## Model
    trainable_modules: Optional[str] = None  # train all parameters if None
    name_lr_mult: Optional[str] = field(default_factory=lambda: "class_embedding,")
    lr_mult: float = 10.
    ### Conditioning
    zero_init_conv_in: bool = True  # whether zero_init new conv_in params
    view_concat_condition: bool = True  # `True` for image-cond
    input_concat_plucker: bool = True
    input_concat_binary_mask: bool = True
    input_concat_warpped_image: bool = False
    
    ### Inference
    init_std: float = 0.  # cf. Instant3D inference trick, `0.` means not used
    init_noise_strength: float = 0.98  # used with `init_std`; cf. Instant3D inference trick, `1.` means not used
    init_bg: float = 0.  # used with `init_std` and `init_noise_strength`; gray background for the initialization

    # Training
    chunk_size: int = 1  # chunk size for GSRecon and GSVAE inference to save memory
    coord_weight: float = 0.  # render coords for supervision
    normal_weight: float = 0.  # render normals for supervision
    recon_weight: float = 0.  # GSVAE reconstruction weight
    render_weight: float = 1.0  # GSVAE rendering weight
    diffusion_weight: float = 1.  # GSDiff diffusion weight
    depth_weight: float = 0.  # render depth for supervision

        ## LPIPS
    lpips_resize: int = 256  # `0` means no resizing
    lpips_weight: float = 1.0  # lpips weight in GSRecon, GSVAE, GSDiff rendering
    lpips_warmup_start: int = 0
    lpips_warmup_end: int = 0



def _update_opt(opt: Options, **kwargs) -> Options:
    new_opt = deepcopy(opt)
    for k, v in kwargs.items():
        setattr(new_opt, k, v)
    return new_opt


# Set all options for different tasks and models
opt_dict: Dict[str, Options] = {}



# MVD

## SD15-based
opt_dict["spatialgen_sd15"] = Options(   
    pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
    prediction_types=["rgb", "depth", "semantic"],
)
## SD21-based
opt_dict["spatialgen_sd21"] = Options(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
    prediction_types=["rgb", "depth", "semantic"],
)
## PAS-based
opt_dict["spatialgen_pas"] = Options(
    prompt_embed_dir="/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/T5_caption_embeds/",
    pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
    prediction_types=["rgb", "depth", "semantic"],
)

# SCM-VAE

## SD21-based
opt_dict["scmvae_sd21"] = Options(
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
    prediction_types=["rgb", "depth"], 
    use_scene_coord_map=True,
    use_metric_depth=False,
)
## SDXL-based
opt_dict["scmvae_pas"] = Options(
    pretrained_model_name_or_path="/seaweedfs/training/experiments/zhenqing/cache/pixart_sigma_sdxlvae_T5_diffusers",
    prediction_types=["rgb", "depth"], 
    use_scene_coord_map=True,
    use_metric_depth=False,
)
