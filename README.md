# SpatialGen: Layout-guided 3D Indoor Scene Generation

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="assets/logo_light.png#gh-light-mode-only" width="60%" alt="SpatialLM" />
  <img src="assets/logo_dark.png#gh-dark-mode-only" width="60%" alt="SpatialLM" />
</div>
<hr style="margin-top: 0; margin-bottom: 8px;">
<div align="center" style="margin-top: 0; padding-top: 0; line-height: 1;">
    <a href="https://manycore-research.github.io/SpatialGen" target="_blank" style="margin: 2px;"><img alt="Project"
    src="https://img.shields.io/badge/🌐%20Project-SpatialGen-ffc107?color=42a5f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://arxiv.org/abs/2509.14981" target="_blank" style="margin: 2px;"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-SpatialGen-b31b1b?logo=arxiv&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://github.com/manycore-research/SpatialGen" target="_blank" style="margin: 2px;"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-SpatialGen-24292e?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://huggingface.co/manycore-research/SpatialGen-1.0" target="_blank" style="margin: 2px;"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpatialGen-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<div align="center">

[Chuan Fang](https://fangchuan.github.io/), [Heng Li](https://hengli.me/), [Yixun Liang](https://yixunliang.github.io/), [Jia Zheng](https://bertjiazheng.github.io), [Yongsen Mao](https://sammaoys.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Rui Tang](https://scholar.google.com/citations?user=dwvfKSkAAAAJ), [Zihan Zhou](https://zihan-z.github.io), [Ping Tan](https://pingtan.people.ust.hk/index.html)

**[HKUST Spatial Artificial Intelligence Lab](https://github.com/HKUST-SAIL)**; **[Manycore Tech Inc](https://manycoretech.com/)**

</div>

<div align="center">

| Image-to-Scene Results                   | Text-to-Scene Results                      |
| :--------------------------------------: | :----------------------------------------: |
| ![Img2Scene](./assets/vis_img2scene.png) | ![Text2Scene](./assets/vis_text2scene.png) |

<p>TL;DR: Given a 3D semantic layout, SpatialGen can generate a 3D indoor scene conditioned on either a reference image (left) or a textual description (right) using a multi-view, multi-modal diffusion model.</p>
</div>

## ✨ News

- [Nov, 2025] SpatialGen is accepted to 3DV 2026!
- [Sep, 2025] We release the paper of SpatialGen!
- [Aug, 2025] Initial release of SpatialGen-1.0!

## 📋 Release Plan

- [x] Provide inference code of SpatialGen.
- [ ] Provide training instruction for SpatialGen.
- [ ] Release SpatialGen dataset.

## SpatialGen Models

<div align="center">

| **Model**                 | **Download**                                                                         |
| :-----------------------: | -------------------------------------------------------------------------------------|
| SpatialGen-1.0            | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialGen-1.0)            |
| FLUX.1-Wireframe-dev-lora | [🤗 HuggingFace](https://huggingface.co/manycore-research/FLUX.1-Wireframe-dev-lora) |

</div>

## Usage

### 🔧 Installation

Tested with the following environment:
* Python 3.10
* PyTorch 2.3.1
* CUDA Version 12.1

```bash
# clone the repository
git clone https://github.com/manycore-research/SpatialGen.git
cd SpatialGen

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Optional: fix the [flux inference bug](https://github.com/vllm-project/vllm/issues/4392)
pip install nvidia-cublas-cu12==12.4.5.8
```

### 📊 Dataset

We provide [SpatialGen-Testset](https://huggingface.co/datasets/manycore-research/SpatialGen-Testset) with 48 rooms, which labeled with 3D layout and 4.8K rendered images (48 x 100 views, including RGB, normal, depth maps and semantic maps) for MVD inference.

### Inference

```bash
# download the pretrain weights
huggingface-cli download --resume-download manycore-research/SpatialGen-1.0 --local-dir spatialgen_ckpts

# Single image-to-3D Scene
bash scripts/infer_spatialgen_i2s.sh

# Text-to-image-to-3D Scene
# step 1. prepare controlnet conditional images, save to /path/to/your/spatialgen-testset/scene_xxxx/condition
python3 preprocess/prepare_flux_ctrlnet_conditions.py --dataset_dir /path/to/your/spatialgen-testset

# step 2. run spatialgen text2scene
# in captions/spatialgen_testset_captions.jsonl, we provide text prompts of different styles for each room, 
# choose a pair of scene_id and prompt to run the text2scene experiment
bash scripts/infer_spatialgen_t2s.sh
```

## License

[SpatialGen-1.0](https://huggingface.co/manycore-research/SpatialGen-1.0) is derived from [Stable-Diffusion-v2.1](https://github.com/Stability-AI/stablediffusion), which is licensed under the [CreativeML Open RAIL++-M License](https://github.com/Stability-AI/stablediffusion/blob/main/LICENSE-MODEL). [FLUX.1-Wireframe-dev-lora](https://huggingface.co/manycore-research/FLUX.1-Wireframe-dev-lora) is licensed under the [FLUX.1-dev Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

## Citation

```bibtex
@inproceedings{SpatialGen,
  title     = {SpatialGen: Layout-guided 3D Indoor Scene Generation},
  author    = {Fang, Chuan and Li, Heng and Liang, Yixu and Zheng, Jia and Mao, Yongsen and Liu, Yuan and Tang, Rui and Zhou, Zihan and Tan, Ping},
  booktitle = {International Conference on 3D Vision},
  year      = {2026}
}
```

## Acknowledgements

We would like to thank the following projects that made this work possible:

[DiffSplat](https://github.com/chenguolin/DiffSplat) | [SD 2.1](https://github.com/Stability-AI/stablediffusion) | [TAESD](https://github.com/madebyollin/taesd) | [FLUX](https://github.com/black-forest-labs/flux/) | [SpatialLM](https://github.com/manycore-research/SpatialLM)
