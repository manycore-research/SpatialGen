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
import sys

sys.path.append(".")
sys.path.append("..")
import os.path as osp
import collections
import math
from pathlib import Path

import torch
import trimesh
import cv2
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation


from src.utils.typing import *
from src.utils.filters import SpatialGradient
from src.utils.cam_ops import (
    get_ray_directions,
    get_rays,
    get_plucker_rays,
    bbox_get_rays,
    opengl_to_opencv,
    unproject_depth,
)
from src.utils.equilib import cube2equi, equi2pers
from src.utils.misc import read_json, readlines
from src.utils.image_utils import crop_image_to_target_ratio
from src.utils.colmap_utils import qvec2rotmat
from src.utils.pcl_ops import cross_view_point_rendering, convert_distance_to_z
from src.data.view_sampler import ViewSampler
from src.utils.layout_utils import (
    compute_intersections,
    parse_obj_bbox_from_meta,
    DEFAULT_UNKNOWN_SEM2COLOR,
    compute_camera_inside_bbox,
    trimesh_to_p3dmesh,
    SDCMeshRenderer,
    parse_spatiallm_obj_bbox_from_meta,
)
from src.utils.vis_util import colorize_depth
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

SubviewImage = collections.namedtuple("SubviewImage", ["id", "image", "yaw_angle", "pitch_angle", "roll_angle"])


def get_koolai_room_ids(data_dir: str, split_filepath: str, invalid_split_filepath: str = None):
    # load valid room ids
    with open(split_filepath, "r") as f:
        uids = f.readlines()
        room_uids = [uid.strip() for uid in uids]

    # load invalid room ids
    if invalid_split_filepath is not None:
        with open(invalid_split_filepath, "r") as f:
            invalid_room_uids = f.readlines()
            invalid_room_uids = [uid.strip() for uid in invalid_room_uids]
    else:
        invalid_room_uids = []

    valid_room_uids = [osp.join(data_dir, uid) for uid in room_uids if uid not in invalid_room_uids]
    return valid_room_uids


def get_spatiallm_room_ids(data_dir: str):
    # valid_scene_filepath = os.path.join(data_dir, "testing_scenes.txt")
    valid_scene_filepath = os.path.join(data_dir, "new_spatiallm_testing_scenes.txt")
    with open(valid_scene_filepath, "r") as f:
        room_uids = f.readlines()
        room_uids = [uid.strip() for uid in room_uids]

    valid_room_uids = [osp.join(data_dir, uid) for uid in room_uids]
    return valid_room_uids


def get_hypersim_room_ids(data_dir: str):
    scene_ids = [ scene_id
        for scene_id in os.listdir(data_dir)
        if os.path.isdir(osp.join(data_dir, scene_id)) and len(os.listdir(osp.join(data_dir, scene_id))) > 1
    ]
    # scene_ids = [scene_id for scene_id in scene_ids if not 'ai_012_009' in scene_id]
    scene_ids = list(set(scene_ids))
    scene_ids =  [osp.join(data_dir, scene_id) for scene_id in scene_ids]
    
    return scene_ids

def get_structured3d_room_ids(data_dir: str):
    split_filepath = os.path.join(data_dir, "testing_scenes.txt")
    scene_ids = readlines(split_filepath)
    scene_paths = [osp.join(data_dir, scene_id) for scene_id in scene_ids]
    return scene_paths

TO_DIFFUSION_TENSOR = torchvision.transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)


class MixDataset(Dataset):
    def __init__(
        self,
        dataset_names: List[str],
        spatialgen_data_dir: str,
        hypersim_data_dir: str,
        structured3d_data_dir: str,
        split_filepath: str,
        image_height: int = 256,
        image_width: int = 256,
        T_in: int = 1,
        total_view: int = 8,
        validation: bool = False,
        sampler_type: str = "random",  # random, sequential, fix
        use_normal: bool = False,
        use_semantic: bool = False,
        use_metric_depth: bool = False,
        use_layout_prior: bool = False,
        use_scene_coord_map: bool = False,
        use_layout_prior_from_p3d: bool = False,
        return_metric_data: bool = True,
    ):
        """This is a MixDataset class that combines KoolAI, ScanNet, and ScanNet++ datasets.
        It is designed to be used for training and evaluation of the Room3D method
        KoolAI Dataset Folder structure:
            data_dir:
            ├── scene_id:
                |-- perspective:
                    |-- room_id:
                        |-- pano_rgb:      # extract perspective image from panorama
                        |-- pano_depth:
                        |-- pano_normal:
                        |-- pano_semantic:

                            ...
                        |-- room_meta.json # room metadata
        ScanNet Dataset Folder structure:
            data_dir
            ├── scans (val and train scans)
                |-- scene0000_00
                    |-- scene0707.txt (scan metadata and intrinsics)
                    |-- scene0707_00_vh_clean_2.ply (gt mesh)
                    |-- rgb
                        |-- frame-0.png
                        |-- frame-1.png
                        |-- ...
                    |-- depth
                        |-- frame-0.png
                        |-- frame-1.png
                        |-- ...
                    |-- pose
                        |-- frame-0.pose.txt  # c2w camera pose
                        |-- frame-1.pose.txt
            ├── scans_test (test scans)
                (same as scans)

        ScanNetPP Dataset Folder structure:
            data_dir
            ├── scene_id
                |-- dslr
                |   `-- nerfstudio
                |-- iphone
                |   |-- colmap
                        |-- images.txt   (camera poses)
                        |-- cameras.txt  (intrinsics)
                |   |-- mask
                |   |-- render_depth  # depth we use
                |   `-- rgb           # rgb we use
                `-- scans
        Args:
            spatialgen_data_dir (str): _description_
            split_filepath (str): _description_
            sequential (_type_): _description_
            invalid_split_filepath (str, optional): _description_. Defaults to None.
            image_height (int, optional): _description_. Defaults to 256.
            image_width (int, optional): _description_. Defaults to 256.
            T_in (int, optional): number of input views. Defaults to 3.
            total_view (int, optional): _description_. Defaults to 8.
            validation (bool, optional): _description_. Defaults to False.
            sampler_type (str, optional): _description_. Defaults to "random".
            fixuse_plucker_ray (bool, optional): _description_. Defaults to True.
            use_metric_depth (bool, optional): whether to use metric depth. Defaults to False.
            use_add_supervision_view (bool, optional): whether to use additional supervision view for GS rendering. Defaults to False.
            return_metric_data (bool, optional): _description_. Defaults to False.
        """
        # support datasets: spatialgen, hypersim, structured3d, spatiallm
        self.load_datasets = dataset_names
        if "spatialgen" in dataset_names and not os.path.exists(spatialgen_data_dir):
            print(f"[Warning] spatialgen_data_dir {spatialgen_data_dir} does not exist, skip loading spatialgen dataset")
            raise ValueError
        if "hypersim" in dataset_names and not os.path.exists(hypersim_data_dir):
            print(f"[Warning] hypersim_data_dir {hypersim_data_dir} does not exist, skip loading hypersim dataset")
            raise ValueError
        if "structured3d" in dataset_names and not os.path.exists(structured3d_data_dir):
            print(f"[Warning] structured3d_data_dir {structured3d_data_dir} does not exist, skip loading structured3d dataset")
            raise ValueError

        self.image_width = image_width
        self.image_height = image_height
        self.spatialgen_data_dir = spatialgen_data_dir
        self.samples = []
        
        self.T_in = T_in if T_in is not None else 3
        self.num_sample_views = total_view
        self.cam_sample_type = sampler_type
        self.use_normal = use_normal
        self.use_semantic = use_semantic
        self.return_metric_data = return_metric_data
        assert not (
            use_metric_depth and use_scene_coord_map
        ), "use_metric_depth and use_scene_coord_map cannot be used together"
        self.use_metric_depth = use_metric_depth
        self.use_scene_coord_map = use_scene_coord_map
        self.sample_consecutive_views = False
        self.use_layout_prior = use_layout_prior
        self.use_layout_prior_from_p3d = use_layout_prior_from_p3d


        self.DEPTH_MAX = 12.593009948730469
        self.DEPTH_MIN = 0.05

        self.yaw_increase_angle = 60.0
        self.is_validation = validation
        if "spatialgen" in self.load_datasets:
            # get spatialgen rooms
            spatialgen_samples = get_koolai_room_ids(spatialgen_data_dir, split_filepath)
            # take the first 30 samples for validation
            if validation:
                spatialgen_samples = spatialgen_samples[-50:]
            else:
                spatialgen_samples = spatialgen_samples[:-50]

            self.samples.extend([{'spatialgen': s} for s in spatialgen_samples])

        if "hypersim" in self.load_datasets:
            # get hypersim rooms
            hypersim_samples = get_hypersim_room_ids(hypersim_data_dir)
            # take the same samples for validation as SceneCraft
            if validation:
                hypersim_samples = [os.path.join(hypersim_data_dir, s) for s in ["ai_010_004", "ai_010_005"]]
            else:
                # repeat scannet samples to achieve the same number of samples as spatialgen
                hypersim_samples = hypersim_samples * 100

            self.samples.extend([{'hypersim': s} for s in hypersim_samples])
            
        if "structured3d" in self.load_datasets:
            st3d_samples = get_structured3d_room_ids(structured3d_data_dir)
            # take the same samples for validation as Ctrl-Room
            if validation:
                st3d_samples = [os.path.join(structured3d_data_dir, s) for s in [ "scene_03356_584", "scene_03110_207", "scene_03453_24234", "scene_03011_221", "scene_03089_576", "scene_03490_1017967", "scene_03324_1097", "scene_03016_266457", "scene_03026_793723", "scene_03300_190736"]]
            else:
                st3d_samples = st3d_samples

            self.samples.extend([{'structured3d': s} for s in st3d_samples])


        self.gradient_checker = SpatialGradient(mode="sobel", order=1, normalized=True)

        sem2color_filepath = os.path.join(self.spatialgen_data_dir, "sem2color_adek150.json")
        self.sem2color_dict = read_json(sem2color_filepath)
        self.color2sem_dict = {tuple(v["color"]): k for k, v in self.sem2color_dict.items()}

        if self.use_layout_prior_from_p3d:
            self.cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self.p3d_renderer = SDCMeshRenderer(
                cameras=None, image_size=(self.image_width, self.image_height), device=self.cuda_device
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_pil_image(file_path: str) -> Float[Tensor, "1 C H W"]:
        """Load image , Pillow version"""
        img = Image.open(file_path)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis].astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def read_image_file(
        img: Image,
        height=None,
        width=None,
        value_scale_factor=1.0,
        resampling_mode=Image.BILINEAR,
        disable_warning=False,
        target_aspect_ratio=None,
    ):
        """ " Reads an image file using PIL, then optionally resizes the image,
        with selective resampling, scales values, and returns the image as a
        tensor

        Args:
            filepath: path to the image.
            height, width: resolution to resize the image to. Both must not be
                None for scaling to take place.
            value_scale_factor: value to scale image values with, default is 1.0
            resampling_mode: resampling method when resizing using PIL. Default
                is PIL.Image.BILINEAR
            target_aspect_ratio: if not None, will crop the image to match this
            aspect ratio. Default is None

        Returns:
            img: tensor with (optionally) scaled and resized image data.

        """

        if target_aspect_ratio:
            img = crop_image_to_target_ratio(img, target_aspect_ratio)

        # resize if both width and height are not none.
        if height is not None and width is not None:
            img_width, img_height = img.size
            # do we really need to resize? If not, skip.
            if (img_width, img_height) != (width, height):
                # warn if it doesn't make sense.
                if (width > img_width or height > img_height) and not disable_warning:
                    print(
                        f"WARNING: target size ({width}, {height}) has a "
                        f"dimension larger than input size ({img_width}, "
                        f"{img_height})."
                    )
                img = img.resize((width, height), resample=resampling_mode)

        img = torchvision.transforms.functional.to_tensor(img).float() * value_scale_factor

        return img

    @staticmethod
    def _load_koolai_camera_pose(camera_meta_dict: Dict, idx: int) -> torch.Tensor:
        """load c2w pose from camera meta

        Args:
            camera_meta_dict (Dict): camera meta json
            idx (int): camera idx

        Returns:
            torch.Tensor: pose
        """
        cam_meta = camera_meta_dict[str(idx)]

        # w2c pose
        pose = np.array(cam_meta["camera_transform"]).reshape(4, 4)
        c2w_pose = np.linalg.inv(pose)
        c2w_pose = torch.from_numpy(c2w_pose).float()
        return c2w_pose

    def _load_koolai_valid_frame_ids(self, room_folderpath: str) -> List[str]:
        """load valid frame ids from room folder

        Args:
            room_folderpath (str): room folder path

        Returns:
            List[str]: valid frame ids
        """
        valid_frames_filepath = osp.join(room_folderpath, "valid_frames.txt")
        valid_frame_ids = [frame_id for frame_id in readlines(valid_frames_filepath) if frame_id.strip() != ""]
        valid_frame_ids.sort(key=lambda x: int(x))
        return valid_frame_ids

    def is_valid_spiral_frame_exist(self, room_folderpath: str) -> bool:
        """check if valid spiral frame exist in the room folder

        Args:
            room_folderpath (str): room folder path

        Returns:
            bool: True if valid spiral frame exist
        """
        valid_frames_filepath = osp.join(room_folderpath, "valid_spiral_frames.txt")
        return osp.exists(valid_frames_filepath)

    def _load_koolai_spiral_frame_ids(self, room_folderpath: str) -> List[str]:
        """load valid frame ids for spiral camera trajectory from room folder

        Args:
            room_folderpath (str): room folder path

        Returns:
            List[str]: valid frame ids
        """
        valid_frames_filepath = osp.join(room_folderpath, "valid_spiral_frames.txt")
        valid_frame_ids = [frame_id for frame_id in readlines(valid_frames_filepath) if frame_id.strip() != ""]
        valid_frame_ids.sort(key=lambda x: int(x))
        return valid_frame_ids

    def convert_distance_to_z(
        self, distance_map: Float[Tensor, "1 1 H W"], focal_length: float = 1.0
    ) -> Float[Tensor, "1 1 H W"]:
        """
        Convert distance map to depth map
        params:
            distance_map: [1, 1, H, W]
            focal_length: noormlized focal length
        """

        vs, us = torch.meshgrid(
            torch.linspace(-1, 1, self.image_height), torch.linspace(-1, 1, self.image_width), indexing="ij"
        )
        us = us.reshape(1, 1, self.image_height, self.image_width)
        vs = vs.reshape(1, 1, self.image_height, self.image_width)
        depth_cos = torch.cos(torch.atan2(torch.sqrt(us * us + vs * vs), torch.tensor(focal_length)))
        depth = distance_map * depth_cos

        return depth

    def check_image_gradient(self, image: Float[Tensor, "1 3 H W"], thresh: float = 0.2) -> bool:
        # compute the gradient, [-0.5, 0.5]
        grad: Float[Tensor, "1 3 2 H W"] = self.gradient_checker(image)
        grad = grad.squeeze(0) - grad.min()
        return grad.mean() > thresh

    def cubemap_to_persepctive_images(
        self,
        equi_img: Float[Tensor, "1 C H W"],
        subview_yaws_lst: List[float],
        subview_pitchs_lst: List[float],
        subview_rolls_lst: List[float],
        subview_fov_x: float = 90.0,
        image_type: str = "rgb",
        subview_img_resolution: int = 256,
        depth_scale: float = 4000.0,
        normalized_focal_length: float = 1.0,
        disable_img_grad_check: bool = False,
    ) -> List[Float[Tensor, "1 C H W"]]:
        """
        Convert equirectangular image to perspective images
        Args:
            equi_img: pre-cached equirectangular image
            subview_yaws_lst: yaw angles of subviews
            subview_pitchs_lst: pitch angles of subviews
            subview_rolls_lst: roll angles of subviews
            subview_fov_x: fov of subviews
            image_type: image type, rgb, depth, normal, semantic
            subview_img_resolution: resolution of subviews
            depth_scale: scale of depth
            normalized_focal_length: normalized focal length

        Returns:
            List[Float[Tensor, "1 C H W"]]: list of perspective images
        """
        subview_img_lst = []
        interpolation_mode = "nearest" if image_type in ["depth", "normal"] else "bilinear"
        rotations_dict_lst = [
            {"roll": roll * np.pi / 180.0, "pitch": -pitch * np.pi / 180.0, "yaw": -yaw * np.pi / 180.0}
            for yaw, pitch, roll in zip(subview_yaws_lst, subview_pitchs_lst, subview_rolls_lst)
        ]
        num_pers_imgs = len(rotations_dict_lst)
        if image_type == "rgb":
            pers_img = equi2pers(
                equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
                rots=rotations_dict_lst,
                height=512,
                width=512,
                fov_x=subview_fov_x,
                mode=interpolation_mode,
            )  # [8, 3, 256, 256]
            if pers_img.shape[-2:] != (subview_img_resolution, subview_img_resolution):
                pers_img = [
                    torch.from_numpy(
                        cv2.resize(
                            img.permute(1, 2, 0).cpu().numpy(),
                            (subview_img_resolution, subview_img_resolution),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    ).permute(2, 0, 1)[None, :, :, :]
                    for img in pers_img
                ]
                pers_img = torch.cat(pers_img, dim=0)
            pers_img = pers_img.float() / 255.0
            pers_img = pers_img.clip(0.0, 1.0)
            if not disable_img_grad_check:
                # only keep the first image that has gradient
                for idx in range(num_pers_imgs):
                    if self.check_image_gradient(pers_img[idx : idx + 1]):
                        subview_img_lst.append(
                            SubviewImage(
                                id=idx,
                                image=pers_img[idx : idx + 1],
                                yaw_angle=subview_yaws_lst[idx],
                                pitch_angle=subview_pitchs_lst[idx],
                                roll_angle=subview_rolls_lst[idx],
                            )
                        )
                        break
            else:
                subview_img_lst = [
                    SubviewImage(
                        id=0,
                        image=pers_img[0:1],
                        yaw_angle=subview_yaws_lst[0],
                        pitch_angle=subview_pitchs_lst[0],
                        roll_angle=subview_rolls_lst[0],
                    )
                ]

        elif image_type == "depth":
            pers_img = equi2pers(
                equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
                rots=rotations_dict_lst,
                height=512,
                width=512,
                fov_x=subview_fov_x,
                mode=interpolation_mode,
            )  # [1, 1, 256, 256]
            if pers_img.shape[-2:] != (subview_img_resolution, subview_img_resolution):
                pers_img = [
                    torch.from_numpy(
                        cv2.resize(
                            img.permute(1, 2, 0).cpu().numpy().squeeze(),
                            (subview_img_resolution, subview_img_resolution),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )[None, None, :, :]
                    for img in pers_img
                ]
                pers_img = torch.cat(pers_img, dim=0)
            pers_img = pers_img.float() / depth_scale
            pers_img = self.convert_distance_to_z(distance_map=pers_img, focal_length=normalized_focal_length)
            subview_img_lst = [
                SubviewImage(
                    id=idx,
                    image=pers_img[idx : idx + 1],
                    yaw_angle=subview_yaws_lst[idx],
                    pitch_angle=subview_pitchs_lst[idx],
                    roll_angle=subview_rolls_lst[idx],
                )
                for idx in range(num_pers_imgs)
            ]

        elif image_type == "normal":
            pers_imgs = equi2pers(
                equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
                rots=rotations_dict_lst,
                height=512,
                width=512,
                fov_x=subview_fov_x,
                mode=interpolation_mode,
            )  # [1, 3, 256, 256]
            if pers_imgs.shape[-2:] != (subview_img_resolution, subview_img_resolution):
                pers_imgs = [
                    torch.from_numpy(
                        cv2.resize(
                            img.permute(1, 2, 0).cpu().numpy(),
                            (subview_img_resolution, subview_img_resolution),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    ).permute(2, 0, 1)[None, :, :, :]
                    for img in pers_imgs
                ]
                pers_imgs = torch.cat(pers_imgs, dim=0)
            for idx in range(num_pers_imgs):
                # convert normal to caemra space
                pers_img = pers_imgs[idx].permute(1, 2, 0).cpu().numpy()
                # convert rgb normal to [-1, 1]
                normal = np.clip((pers_img + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
                normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)
                # rotate reference view normal to subview
                R_z = Rotation.from_euler("z", rotations_dict_lst[idx]["roll"], degrees=False).as_matrix()
                R_x = Rotation.from_euler("x", -rotations_dict_lst[idx]["pitch"], degrees=False).as_matrix()
                R_y = Rotation.from_euler("y", -rotations_dict_lst[idx]["yaw"], degrees=False).as_matrix()
                R_ref_subview = R_y @ R_x @ R_z
                rot_normal = (normal.reshape(-1, 3) @ R_ref_subview).reshape(normal.shape)
                # save normal in camera space, flip to make +z upward
                pers_img = torch.from_numpy(rot_normal).permute(2, 0, 1).float()
                pers_img = pers_img.unsqueeze(0)  # [1, 3, 256, 256]
                subview_img_lst.append(
                    SubviewImage(
                        id=idx,
                        image=pers_img,
                        yaw_angle=subview_yaws_lst[idx],
                        pitch_angle=subview_pitchs_lst[idx],
                        roll_angle=subview_rolls_lst[idx],
                    )
                )

        elif image_type == "semantic":
            pers_img = equi2pers(
                equi=equi_img.repeat(num_pers_imgs, 1, 1, 1),
                rots=rotations_dict_lst,
                height=512,
                width=512,
                fov_x=subview_fov_x,
                mode=interpolation_mode,
            )  # [1, 3, 256, 256]
            pers_img = pers_img.float() / 255.0
            if pers_img.shape[-2:] != (subview_img_resolution, subview_img_resolution):
                pers_img = [
                    torch.from_numpy(
                        cv2.resize(
                            img.permute(1, 2, 0).cpu().numpy(),
                            (subview_img_resolution, subview_img_resolution),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    ).permute(2, 0, 1)[None, :, :, :]
                    for img in pers_img
                ]
                pers_img = torch.cat(pers_img, dim=0)
            subview_img_lst = [
                SubviewImage(
                    id=idx,
                    image=pers_img[idx : idx + 1],
                    yaw_angle=subview_yaws_lst[idx],
                    pitch_angle=subview_pitchs_lst[idx],
                    roll_angle=subview_rolls_lst[idx],
                )
                for idx in range(num_pers_imgs)
            ]
        # subview_img_lst.append(SubviewImage(id=idx, image=pers_img, yaw_angle=yaw_degree, pitch_angle=pitch_degree, roll_angle=roll_degree))

        return subview_img_lst

    def depth_scale_shift_normalization(
        self,
        depths: Float[Tensor, "N 1 H W"],
        valid_mask: Float[Tensor, "N 1 H W"] = None,
        min: float = None,
        max: float = None,
        move_invalid_to_far_plane: bool = True,
    ) -> Tuple[Float[Tensor, "N 1 H W"], float, float]:
        """
        scale depth to [-1, 1], use the same scale for all depth maps
        params:
            depths: [N, 1, H, W]
        Returns:
            normalized_depths: [N, 1, H, W]
            max_value: max value of depth maps
        """

        if valid_mask is None:
            valid_mask = torch.ones_like(depths).bool()
        valid_mask = valid_mask & (depths > 0)

        # # normalize to [-1, 1]
        if min is not None or max is not None:
            min_value, max_value = torch.tensor([min], dtype=torch.float32), torch.tensor([max], dtype=torch.float32)
        else:
            min_value, max_value = depths.min(), depths.max()
        # print(f"depth_min: {min_value}, depth_max: {max_value}")
        normalized_depths = (depths - min_value) / (max_value - min_value) * 2.0 - 1.0
        normalized_depths = torch.clip(normalized_depths, -1.0, 1.0)

        normalized_depths[~valid_mask] = 1.0 if move_invalid_to_far_plane else -1.0
        return normalized_depths, min_value, max_value

    @staticmethod
    def descale_scale_shift_normalization(
        depths: Float[Tensor, "N 1 H W"], min_value: float, max_value: float
    ) -> Float[Tensor, "N 1 H W"]:
        """
        descale depth to original value
        params:
            depths: [N, 1, H, W]
            min_value: min value of depth maps
            max_value: max value of depth maps
        Returns:
            descaled_depths: [N, 1, H, W]
        """
        denormalized_depths = (depths + 1.0) / 2.0 * (max_value - min_value) + min_value
        return denormalized_depths

    def metric_depth_normalization(
        self,
        depths: Float[Tensor, "N 1 H W"],
        valid_mask: Float[Tensor, "N 1 H W"] = None,
        move_invalid_to_far_plane: bool = True,
    ) -> Tuple[Float[Tensor, "N 1 H W"], float, float]:
        """
        scale metric depth to [-1, 1], use the metric depth maps
        params:
            depths: [N, 1, H, W]
        Returns:
            normalized_depths: [N, 1, H, W]
            max_value: max value of depth maps
            pose_scale: 1./scale of the scene
        """

        # Ref: https://github.com/AnshShah3009/MetricGold

        if valid_mask is None:
            valid_mask = torch.ones_like(depths).bool()
        valid_mask = valid_mask & (depths > 0)

        # # normalize to [-1, 1]
        min_value, max_value = torch.tensor([self.DEPTH_MIN], dtype=torch.float32), torch.tensor(
            [self.DEPTH_MAX], dtype=torch.float32
        )
        log_depths = torch.log(depths / min_value) / torch.log(max_value / min_value)
        normalized_depths = log_depths * 2.0 - 1.0
        normalized_depths = torch.clip(normalized_depths, -1.0, 1.0)

        normalized_depths[~valid_mask] = 1.0 if move_invalid_to_far_plane else -1.0
        return normalized_depths, min_value, max_value

    @staticmethod
    def descale_metric_log_normalization(
        depths: Float[Tensor, "N 1 H W"], abs_min_value: float, abs_max_value: float
    ) -> Float[Tensor, "N 1 H W"]:
        """
        descale metric depth to original value
        params:
            depths: [N, 1, H, W]
            min_value: min value of depth maps
            max_value: max value of depth maps

        Returns:
            descaled_depths: [N, 1, H, W]
        """
        denormalized_depths = (depths + 1.0) / 2.0
        if type(abs_min_value) != torch.Tensor:
            abs_min_value = torch.tensor(abs_min_value, dtype=torch.float32)
        if type(abs_max_value) != torch.Tensor:
            abs_max_value = torch.tensor(abs_max_value, dtype=torch.float32)
        descaled_depths = torch.exp(denormalized_depths * torch.log(abs_max_value / abs_min_value)) * abs_min_value
        return descaled_depths

    def calc_scale_mat(
        self, raw_poses: Float[Tensor, "N 4 4"], depth_range: float = 1.5, offset_center: bool = True
    ) -> Tuple[Float[Tensor, "4 4"], float]:
        """
        raw_poses: [N, 4, 4], camera to world poses
        depth_range: maximum depth range of each camera
        """
        c2w_poses = raw_poses
        min_vertices = c2w_poses[:, :3, 3].min(dim=0).values
        max_vertices = c2w_poses[:, :3, 3].max(dim=0).values

        if offset_center:
            center = (min_vertices + max_vertices) / 2.0
        else:
            center = torch.zeros(3, dtype=torch.float32)

        # convert the scene to [-1, 1] unit cube
        # scale = 2. / (torch.max(max_vertices - min_vertices) + 2 * depth_range)
        # TODO: use above scale, but for now, use the depth scale
        positions_scale = torch.max(max_vertices - min_vertices)
        if depth_range is not None:
            # scale = 2.0 / positions_scale if positions_scale > (2 * depth_range) else 2.0 / (2 * depth_range)
            scale = 2.0 / (2 * depth_range)
        else:
            scale = 2.0 / positions_scale

        # we should normalized to unit cube
        scale_mat = torch.eye(4, dtype=torch.float32)
        scale_mat[:3, 3] = -center
        scale_mat[:3] *= scale

        return scale_mat, scale

    def calc_subview_pose(
        self,
        T_c2w: Float[Tensor, "4 4"],
        T_cv_enu: Float[Tensor, "4 4"],
        yaw_angle: Float[Tensor, "..."],
        pitch_angle: Float[Tensor, "..."],
        roll_angle: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        """Calculate camera poses from room metadata

        Args:
            T_cv_enu: transformation matrix from cv to enu coordinate
            T_c2w: camera pose in enu coordinate
            rot_angle: rotation angle in degree

        Returns:
            Float[Tensor, "camera entry"]: camera pose
        """

        subview_pose = torch.eye(4).float()
        roll_angle = torch.deg2rad(torch.tensor([roll_angle]))
        pitch_angle = torch.deg2rad(torch.tensor([pitch_angle]))
        yaw_angle = torch.deg2rad(torch.tensor([yaw_angle]))
        R_z = torch.tensor(
            [
                [torch.cos(roll_angle), -torch.sin(roll_angle), 0],
                [torch.sin(roll_angle), torch.cos(roll_angle), 0],
                [0, 0, 1],
            ]
        )
        R_x = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(pitch_angle), -torch.sin(pitch_angle)],
                [0, torch.sin(pitch_angle), torch.cos(pitch_angle)],
            ]
        )
        R_y = torch.tensor(
            [
                [torch.cos(yaw_angle), 0, -torch.sin(yaw_angle)],
                [0, 1, 0],
                [torch.sin(yaw_angle), 0, torch.cos(yaw_angle)],
            ]
        )
        subview_pose[:3, :3] = R_y @ R_x @ R_z
        subview_pose = T_c2w @ T_cv_enu @ subview_pose
        return subview_pose

    def prepare_ray_directions(
        self, num_sample_views: int = 16, focal_length: float = 128.0
    ) -> Float[Tensor, "B H W 3"]:
        # get ray directions by intrinsics
        directions = get_ray_directions(
            H=self.image_height,
            W=self.image_width,
            focal=focal_length,
        )
        directions: Float[Tensor, "B H W 3"] = directions[None, :, :, :].repeat(num_sample_views, 1, 1, 1)
        return directions

    def set_num_input_views(self, num_input_views: int):
        self.T_in = num_input_views
        self.T_out = self.num_sample_views - num_input_views

    def set_cam_sampler(self, cam_sample_type: str):
        self.cam_sample_type = cam_sample_type

    def set_sample_consecutive_views(self, sample_consecutive_views: bool):
        self.sample_consecutive_views = sample_consecutive_views

    def get_koolai_valid_cameras(self, sample_room_path: str) -> Dict:
        # sample_room_path = self.samples[index]
        room_uid = "/".join(sample_room_path.split("/")[-4:])
        room_meta_filepath = os.path.join(sample_room_path, "room_meta.json")
        # load camera poses
        room_metadata = read_json(room_meta_filepath)
        cameras_meta = room_metadata["cameras"]
        total_pano_view_keys = self._load_koolai_valid_frame_ids(sample_room_path)

        # random the num of input views
        T_in = self.T_in
        T_out = self.num_sample_views - T_in

        view_sampler = ViewSampler(
            room_metadata=room_metadata,
            num_input_views=T_in,
            num_target_views=T_out,
            yaw_interval_thresh=60.0,
            distance_interval_thresh=1.0,
            trajectory_type=self.cam_sample_type,
        )
        valid_camera_keys = view_sampler.get_valid_camera_keys(
            valid_frames=total_pano_view_keys,
            cameras_dict=cameras_meta,
            room_path=sample_room_path,
        )
        return valid_camera_keys

    def get_koolai_layout(self, sample_room_path: str, return_mesh=False, noceil_mesh=False) -> Dict:
        """parse 3D object layout in the sampled room

        Args:
            sample_room_path (str): room data path

        Returns:
            Dict: dict of object layout
        """
        apartment_id = sample_room_path.split("/")[-3]
        room_meta_filepath = os.path.join(sample_room_path, "room_meta.json")
        data_dir = "/".join(sample_room_path.split("/")[:-4])
        instance2semlabel_filepath = os.path.join(data_dir, "inst2segm", f"{apartment_id}.json")
        # load camera poses
        room_metadata = read_json(room_meta_filepath)
        instid2semid_metadata = read_json(instance2semlabel_filepath)

        return parse_obj_bbox_from_meta(
            room_metadata,
            instid2semid_metadata,
            room_folderpath=sample_room_path,
            return_mesh=return_mesh,
            noceil_mesh=noceil_mesh,
        )

    def get_koolai_item(
        self,
        sample_room_path: str | None = None,
        depth_scale: float = 4000.0,
    ) -> Dict:
        # sample_room_path = osp.join(self.spatialgen_data_dir, "20241118/3FO4K5FWX2T2/perspective/room_788")
        room_uid = "/".join(sample_room_path.split("/")[-4:])
        rgb_dir = os.path.join(sample_room_path, "pano_rgb")
        depth_dir = os.path.join(sample_room_path, "pano_depth")
        normal_dir = os.path.join(sample_room_path, "pano_normal")
        semantic_dir = os.path.join(sample_room_path, "pano_semantic")
        room_meta_filepath = os.path.join(sample_room_path, "room_meta.json")

        # load camera poses
        room_metadata = read_json(room_meta_filepath)
        cameras_meta = room_metadata["cameras"]
        T_enu_cv = np.array(room_metadata["T_enu_cv"]).reshape(4, 4)
        T_cv_enu = torch.inverse(torch.from_numpy(T_enu_cv).float())

        if self.is_valid_spiral_frame_exist(sample_room_path) and self.cam_sample_type in [
            "spiral",
            "panoramic",
            "randomwalk",
        ]:
            # use the filtered panoramic views, which are not too close to each furniture item
            total_pano_view_keys = self._load_koolai_spiral_frame_ids(sample_room_path)
        else:
            # use the whole panoramic views
            total_pano_view_keys = self._load_koolai_valid_frame_ids(sample_room_path)

        # random the num of input views
        T_in = self.T_in
        T_out = self.num_sample_views - T_in

        view_sampler = ViewSampler(
            room_metadata=room_metadata,
            num_sample_views=self.num_sample_views,
            yaw_interval_thresh=60.0,
            distance_interval_thresh=1.0,
            trajectory_type=self.cam_sample_type,
            is_validation=self.is_validation,
        )
        sample_dict = view_sampler.sample(
            valid_frames=total_pano_view_keys,
            cameras_dict=cameras_meta,
            b_make_sampled_views_consecutive=self.sample_consecutive_views,
            room_uid=room_uid,
        )
        sample_keys = sample_dict["sample_keys"]  # sampled view keys
        sample_yaws = sample_dict["sample_yaws"]  # sampled yaw angles, [0, 360)
        sample_rolls = sample_dict["sample_rolls"]  # sampled roll angles, [-15, 15]
        sample_pitches = sample_dict["sample_pitches"]  # sampled pitch angles, [-30, 30]
        # c2w poses, rgbs, background colors
        poses, rgbs = [], []
        depths, normals, semantics = [], [], []

        # TODO: make fov_x random
        random_fov_x = np.random.uniform(45.0, 90.0)  # random fov_x in degree
        # random_fov_x = 55.0  # fixed fov_x, same as SpatialLM-Testset
        random_fov_x = 90.0
        focal_len = 0.5 * self.image_width / math.tan(random_fov_x * math.pi / 360.0)
        normalized_focal_len = 2.0 * focal_len / self.image_width

        intrinsic_mat = torch.tensor(
            [
                [focal_len, 0, self.image_width / 2.0],
                [0, focal_len, self.image_height / 2.0],
                [0, 0, 1],
            ]
        ).float()

        for idx, view_key in enumerate(sample_keys):
            subview_yaw_degree = sample_yaws[idx]

            # load c2w pose
            c2w_pose = self._load_koolai_camera_pose(cameras_meta, view_key)
            # load equirectangular images
            rgb_pano_image_filepath = os.path.join(rgb_dir, f"{view_key}.png")
            depth_pano_image_filepath = os.path.join(depth_dir, f"{view_key}.png")
            normal_pano_image_filepath = os.path.join(normal_dir, f"{view_key}.png")
            semantic_pano_image_filepath = os.path.join(semantic_dir, f"{view_key}.png")

            equi_rgb: Float[Tensor, "1 3 H W"] = self._load_pil_image(rgb_pano_image_filepath)
            equi_depth: Float[Tensor, "1 1 H W"] = self._load_pil_image(depth_pano_image_filepath)
            # subview degrees
            subview_yaw_degrees = [subview_yaw_degree]  # [0, 360]
            subview_pitch_degrees = [sample_pitches[idx]]
            subview_roll_degress = [sample_rolls[idx]]

            rgb_subview_lst = self.cubemap_to_persepctive_images(
                equi_img=equi_rgb,
                subview_fov_x=random_fov_x,
                subview_yaws_lst=subview_yaw_degrees,
                subview_pitchs_lst=subview_pitch_degrees,
                subview_rolls_lst=subview_roll_degress,
                subview_img_resolution=self.image_width,
                image_type="rgb",
                disable_img_grad_check=True,  # check the gradient of the image
            )

            selected_subview_yaws = [rgb.yaw_angle for rgb in rgb_subview_lst]
            selected_subview_pitchs = [rgb.pitch_angle for rgb in rgb_subview_lst]
            selected_subview_rolls = [rgb.roll_angle for rgb in rgb_subview_lst]

            # assume len(selected_subview_yaws) is always 1
            depth_subview_lst = self.cubemap_to_persepctive_images(
                equi_img=equi_depth,
                subview_fov_x=random_fov_x,
                subview_yaws_lst=selected_subview_yaws,
                subview_pitchs_lst=selected_subview_pitchs,
                subview_rolls_lst=selected_subview_rolls,
                subview_img_resolution=self.image_width,
                depth_scale=depth_scale,
                image_type="depth",
                normalized_focal_length=normalized_focal_len,
            )
            if self.use_normal:
                equi_normal: Float[Tensor, "1 3 H W"] = self._load_pil_image(normal_pano_image_filepath)
                normal_subview_lst = self.cubemap_to_persepctive_images(
                    equi_img=equi_normal,
                    subview_fov_x=random_fov_x,
                    subview_yaws_lst=selected_subview_yaws,
                    subview_pitchs_lst=selected_subview_pitchs,
                    subview_rolls_lst=selected_subview_rolls,
                    subview_img_resolution=self.image_width,
                    image_type="normal",
                )
            if self.use_semantic:
                equi_semantic: Float[Tensor, "1 3 H W"] = self._load_pil_image(semantic_pano_image_filepath)
                semantic_subview_lst = self.cubemap_to_persepctive_images(
                    equi_img=equi_semantic,
                    subview_fov_x=random_fov_x,
                    subview_yaws_lst=selected_subview_yaws,
                    subview_pitchs_lst=selected_subview_pitchs,
                    subview_rolls_lst=selected_subview_rolls,
                    subview_img_resolution=self.image_width,
                    image_type="semantic",
                )

            # scale rgb to [-1, 1]
            subview_rgbs = [rgb.image * 2.0 - 1.0 for rgb in rgb_subview_lst]
            subview_depths = [depth.image for depth in depth_subview_lst]
            if self.use_normal:
                subview_normals = [normal.image for normal in normal_subview_lst]
            if self.use_semantic:
                subview_semantics = [semantic.image * 2.0 - 1.0 for semantic in semantic_subview_lst]
            subview_poses = [
                self.calc_subview_pose(c2w_pose, T_cv_enu, -rgb.yaw_angle, rgb.pitch_angle, rgb.roll_angle)
                for rgb in rgb_subview_lst
            ]

            poses += subview_poses
            rgbs += subview_rgbs
            depths += subview_depths
            if self.use_normal:
                normals += subview_normals
            if self.use_semantic:
                semantics += subview_semantics

        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.cat(rgbs, dim=0)
        depths: Float[Tensor, "N 1 H W"] = torch.cat(depths, dim=0)
        if self.use_normal:
            normals: Float[Tensor, "N 3 H W"] = torch.cat(normals, dim=0)
        if self.use_semantic:
            semantics: Float[Tensor, "N 3 H W"] = torch.cat(semantics, dim=0)

        total_sample_views = self.num_sample_views
        if poses.shape[0] < total_sample_views:
            pad_num = total_sample_views - poses.shape[0]
            pad_indices = np.random.choice(poses.shape[0], pad_num, replace=True)
            rgbs = torch.cat([rgbs, rgbs[pad_indices]], dim=0)
            depths = torch.cat([depths, depths[pad_indices]], dim=0)
            if self.use_normal:
                normals = torch.cat([normals, normals[pad_indices]], dim=0)
            if self.use_semantic:
                semantics = torch.cat([semantics, semantics[pad_indices]], dim=0)
            poses = torch.cat([poses, poses[pad_indices]], dim=0)
        elif poses.shape[0] > total_sample_views:
            sample_indices = np.random.choice(poses.shape[0], total_sample_views, replace=False)
            rgbs = rgbs[sample_indices]
            depths = depths[sample_indices]
            if self.use_normal:
                normals = normals[sample_indices]
            if self.use_semantic:
                semantics = semantics[sample_indices]
            poses = poses[sample_indices]

        metric_poses = poses.clone()

        # always choose the first view as the reference view
        relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
        # update poses to relative poses
        poses = relative_poses

        if self.use_scene_coord_map:
            # project depth to point cloud
            scene_coord_maps = unproject_depth(depth_map=depths, C2W=poses, K=intrinsic_mat)
            # normalize scene coord maps to [-1, 1]
            coord_max, coord_min = scene_coord_maps.max(), scene_coord_maps.min()
            # max_depth is scene range
            scene_range = torch.abs(coord_max - coord_min)
            scene_coord_maps: Float[Tensor, "N 3 H W"] = (scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
            scene_coord_maps = scene_coord_maps.clamp(-1.0, 1.0)

        obj_layout_dict, obj_mesh_list = self.get_koolai_layout(sample_room_path, return_mesh=True)

        # if self.use_layout_prior:
        #     render_semantics = []
        #     render_depths = []
        #     for i in range(self.num_sample_views):
        #         ray_o, ray_d = bbox_get_rays(self.image_height, self.image_width, intrinsic_mat, metric_poses[i], normalize_dir=False)
        #         # rendered_label_img: [H, W], element: semantic_label_indices: [0, 65535], empty area is 0
        #         render_label_img, render_index_img, render_depth_img = compute_intersections(ray_o,
        #                                                                     ray_d,
        #                                                                     metas=obj_layout_dict,
        #                                                                     height=self.image_height,
        #                                                                     width=self.image_width,
        #                                                                     rays_chunk=8192)

        #         # update semantic label and depth for the fllor, wall and ceiling areas
        #         render_label2sem_color = map(lambda x: self.sem2color_dict.get(str(x), DEFAULT_UNKNOWN_SEM2COLOR)["color"], render_label_img[render_label_img > 0])
        #         render_label2sem_color = np.array(list(render_label2sem_color)).reshape(-1, 3)
        #         semantic_np = ((semantics[i] * 0.5 + 0.5) * 255).permute(1,2,0).clone().cpu().numpy().astype(np.uint8)
        #         render_sem_img = np.zeros_like(semantic_np)
        #         render_sem_img[render_label_img > 0] = render_label2sem_color
        #         render_sem_img[render_label_img == 0] = semantic_np[render_label_img == 0]

        #         render_depth_img = render_depth_img[:,:,None]
        #         depth_np = depths[i].permute(1,2,0).clone().cpu().numpy()
        #         # complete the rendered depth image using the GT depth image, usually the background area: wall, floor, ceiling
        #         render_depth_img[render_depth_img == -1.] = depth_np[render_depth_img == -1.]
        #         # filter out the occluded pixels (when the farther object is occluded by the closer walls)
        #         occluded_area = np.logical_and(render_depth_img > depth_np, depth_np > 0).squeeze()
        #         render_depth_img[occluded_area] = depth_np[occluded_area]
        #         render_sem_img[occluded_area] = semantic_np[occluded_area]

        #         render_semantics.append(TO_DIFFUSION_TENSOR(render_sem_img))
        #         render_depths.append(torch.from_numpy(render_depth_img).permute(2,0,1).float())
        #         # save semantic images
        #         # if True:
        #         #     merged_image = Image.new('RGB', (self.image_width, self.image_height * 3))
        #         #     rgb_image = Image.fromarray(((rgbs[i]*0.5+0.5)*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
        #         #     merged_image.paste(rgb_image, (0, 0))
        #         #     render_sem_img = Image.fromarray(render_sem_img.astype(np.uint8))
        #         #     merged_image.paste(render_sem_img, (0, 1 * self.image_height))
        #         #     merged_image.paste(Image.fromarray(colorize_depth(render_depth_img[:,:,0]).astype(np.uint8)).convert('RGB'), (0, 2 * self.image_height))
        #         #     merged_image.save(f"{room_uid.replace('/', '_')}_{i}_ray.png")
        #         #     rendered_label_img = Image.fromarray(render_label_img.astype(np.int32))
        #         #     rendered_label_img.save(f"{room_uid.replace('/', '_')}_{i}_rendered_label.png")
        #     render_semantics = torch.stack(render_semantics, dim=0)
        #     render_depths = torch.stack(render_depths, dim=0)

        #     if self.use_scene_coord_map:
        #         layout_scene_coord_maps = unproject_depth(depth_map=render_depths, C2W=poses, K=intrinsic_mat)
        #         # normalize scene coord maps to [-1, 1]
        #         layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (layout_scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
        #         layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)
        #     else:
        #         # normalize depth to [-1, 1]
        #         normalized_render_depths, min_render_depth, max_render_depth = self.metric_depth_normalization(render_depths)
        #         # normalized_render_depths = render_depths

        if self.use_layout_prior_from_p3d:
            render_semantics = []
            render_depths = []
            for i in range(self.num_sample_views):
                cam_in_obj_mask = compute_camera_inside_bbox(metric_poses[i], obj_layout_dict)[0]
                pose_device = self.cuda_device
                filtered_mesh_list = [obj_mesh_list[_id] for _id, _x in enumerate(cam_in_obj_mask) if not _x]
                #  add wall mesh
                wall_mesh = obj_mesh_list[-1]
                #  filtered_mesh_list.append(wall_mesh)
                concat_mesh = trimesh.util.concatenate(filtered_mesh_list)
                obj_mesh, face_color = trimesh_to_p3dmesh(concat_mesh)
                obj_mesh = obj_mesh.to(pose_device)
                face_color = face_color.to(pose_device)

                wall_mesh, wall_face_color = trimesh_to_p3dmesh(wall_mesh)
                wall_mesh = wall_mesh.to(pose_device)
                wall_face_color = wall_face_color.to(pose_device)
                w2c = metric_poses[i : i + 1].inverse()

                p3d_camera = cameras_from_opencv_projection(
                    w2c[..., :3, :3],
                    w2c[..., :3, 3],
                    camera_matrix=intrinsic_mat[None,],
                    image_size=torch.tensor([[self.image_height, self.image_width]]),
                ).to(pose_device)

                obj_rener_imgs = self.p3d_renderer(obj_mesh, cameras=p3d_camera, faces_color=face_color)
                wall_render_imgs = self.p3d_renderer(wall_mesh, cameras=p3d_camera, faces_color=wall_face_color)
                wall_render_sem_img = (wall_render_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)
                wall_render_dep_img = wall_render_imgs["depth"].cpu()[0].numpy().astype(np.float32)

                semantic_np = ((semantics[i] * 0.5 + 0.5) * 255).permute(1, 2, 0).clone().cpu().numpy().astype(np.uint8)
                render_sem_img = (obj_rener_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)

                # fill the gt_sem with window color
                black_area = np.all(semantic_np == 0, axis=-1)
                semantic_np[black_area] = np.array([230, 230, 230])
                semantics[i] = TO_DIFFUSION_TENSOR(semantic_np)
                bg_area = np.all(render_sem_img == 0, axis=-1)
                render_sem_img[bg_area] = semantic_np[bg_area]

                depth_np = depths[i].permute(1, 2, 0).clone().cpu().numpy()
                #  fill the gt_depth with all_render_depth
                depth_np[depth_np == 0] = wall_render_dep_img[depth_np == 0]
                depths[i] = torch.from_numpy(depth_np).permute(2, 0, 1).float()

                # fill the rendered depth image using the GT depth image, usually the background area: wall, floor, ceiling
                render_depth_img = obj_rener_imgs["depth"].cpu()[0].numpy()
                render_depth_img[render_depth_img == -1.0] = depth_np[render_depth_img == -1.0]
                # filter out the occluded pixels (when the farther object is occluded by the closer walls)
                occluded_area = np.logical_and(render_depth_img > depth_np, depth_np > 0).squeeze()
                render_depth_img[occluded_area] = depth_np[occluded_area]
                render_sem_img[occluded_area] = semantic_np[occluded_area]

                render_semantics.append(TO_DIFFUSION_TENSOR(render_sem_img))
                render_depths.append(torch.from_numpy(render_depth_img).permute(2, 0, 1).float())

                # save semantic images
            #  if True:
            #      merged_image = Image.new('RGB', (self.image_width, self.image_height * 3))
            #      rgb_image = Image.fromarray(((rgbs[i]*0.5+0.5)*255).permute(1,2,0).cpu().numpy().astype(np.uint8))
            #      merged_image.paste(rgb_image, (0, 0))
            #      render_sem_img = Image.fromarray(render_sem_img.astype(np.uint8))
            #      merged_image.paste(render_sem_img, (0, 1 * self.image_height))
            #      merged_image.paste(Image.fromarray(colorize_depth(render_depth_img[:,:,0]).astype(np.uint8)).convert('RGB'), (0, 2 * self.image_height))
            #      merged_image.save(f"{room_uid.replace('/', '_')}_{i}_pyt3d.png")

            render_semantics = torch.stack(render_semantics, dim=0)
            render_depths = torch.stack(render_depths, dim=0)

            if self.use_scene_coord_map:
                layout_scene_coord_maps = unproject_depth(depth_map=render_depths, C2W=poses, K=intrinsic_mat)
                # normalize scene coord maps to [-1, 1]
                layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (
                    layout_scene_coord_maps - coord_min
                ) / scene_range * 2.0 - 1.0
                layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)
            else:
                # normalize depth to [-1, 1]
                normalized_render_depths, min_render_depth, max_render_depth = self.metric_depth_normalization(
                    render_depths
                )

        metric_depths = depths.clone()

        if not self.use_scene_coord_map:
            # normalize depth to [-1, 1]
            if not self.use_metric_depth:
                normalized_depths, min_depth, max_depth = self.depth_scale_shift_normalization(depths)
            else:
                normalized_depths, min_depth, max_depth = self.metric_depth_normalization(depths)

        if not self.use_metric_depth:
            # normalize poses to unit cube[-1,1] w.r.t current sample views
            curr_scale_mat, curr_scene_scale = self.calc_scale_mat(poses, depth_range=scene_range, offset_center=False)
            for pose_idx in range(poses.shape[0]):
                # scale pose_c2w
                subview_pose = curr_scale_mat @ poses[pose_idx]
                R_c2w = (subview_pose[:3, :3]).numpy()
                q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
                q_c2w = trimesh.transformations.unit_vector(q_c2w)
                R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
                subview_pose[:3, :3] = torch.from_numpy(R_c2w)
                poses[pose_idx] = subview_pose
        else:
            assert not self.use_scene_coord_map
            curr_scene_scale = 1

        # always choose the first view as the reference view
        canonical_ray_directions = self.prepare_ray_directions(
            num_sample_views=total_sample_views, focal_length=focal_len
        )
        rays_o, rays_d = get_rays(canonical_ray_directions, poses, keepdim=True)
        rays_od = torch.cat([rays_o, rays_d], dim=-1)
        rays_od = rays_od.permute(0, 3, 1, 2)  # B, 6, H, W

        # plucker rays
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, H, W, 3
        plucker_rays = torch.cat([rays_dxo, rays_d], dim=-1)  # B, H, W, 6
        plucker_rays = plucker_rays.permute(0, 3, 1, 2)  # B, 6, H, W
        # # update poses to relative poses
        # poses = relative_poses

        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:T_in]

        # atarget views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[T_in : self.num_sample_views]

        if not self.use_scene_coord_map:
            input_depths: Float[Tensor, "N 3 H W"] = normalized_depths[:T_in].repeat(1, 3, 1, 1)
            target_depths: Float[Tensor, "N 3 H W"] = normalized_depths[T_in : self.num_sample_views].repeat(1, 3, 1, 1)
        else:
            input_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[:T_in]
            target_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[T_in : self.num_sample_views]

        if self.use_normal:
            input_normals: Float[Tensor, "N 3 H W"] = normals[:T_in]
            target_normals: Float[Tensor, "N 3 H W"] = normals[T_in : self.num_sample_views]
        if self.use_semantic:
            input_semantics: Float[Tensor, "N 3 H W"] = semantics[:T_in]
            target_semantics: Float[Tensor, "N 3 H W"] = semantics[T_in : self.num_sample_views]

        depth_class = torch.tensor([1, 0, 0, 0]).float()
        depth_task_embeddings = torch.stack([depth_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        color_class = torch.tensor([0, 1, 0, 0]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_normal:
            normal_class = torch.tensor([0, 0, 1, 0]).float()
            normal_task_embeddings = torch.stack([normal_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_semantic:
            semantic_class = torch.tensor([0, 0, 0, 1]).float()
            semantic_task_embeddings = torch.stack([semantic_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            layout_sem_class = torch.tensor([0, 0, 0, 10]).float()
            layout_sem_task_embeddings = torch.stack([layout_sem_class] * self.num_sample_views, dim=0)
            layout_depth_class = torch.tensor([0, 0, 0, 101]).float()
            layout_depth_task_embeddings = torch.stack([layout_depth_class] * self.num_sample_views, dim=0)

        cond_Ts: Float[Tensor, "N 4 4"] = poses[:T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[T_in : self.num_sample_views]

        input_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[:T_in]
        target_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[T_in : self.num_sample_views]

        input_rays: Float[Tensor, "N 6 H W"] = rays_od[:T_in]
        target_rays: Float[Tensor, "N 6 H W"] = rays_od[T_in : self.num_sample_views]

        shuffled_indices = torch.arange(self.num_sample_views)
        input_indices = shuffled_indices[:T_in]
        target_indices = shuffled_indices[T_in:]

        data = {}
        data["dataset"] = "spatialgen"
        data["room_uid"] = room_uid
        data["image_input"] = input_images
        data["image_target"] = target_images
        data["depth_input"] = input_depths
        data["depth_target"] = target_depths
        if self.use_normal:
            data["normal_input"] = input_normals
            data["normal_target"] = target_normals
            data["normal_task_embeddings"] = normal_task_embeddings

        if self.use_semantic:
            data["semantic_input"] = input_semantics
            data["semantic_target"] = target_semantics
            data["semantic_task_embeddings"] = semantic_task_embeddings

        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            data["semantic_layout_input"] = render_semantics[:T_in]
            data["semantic_layout_target"] = render_semantics[T_in : self.num_sample_views]
            if not self.use_scene_coord_map:
                normalized_render_depths = normalized_render_depths.repeat(1, 3, 1, 1)
                data["depth_layout_input"] = normalized_render_depths[:T_in]
                data["depth_layout_target"] = normalized_render_depths[T_in : self.num_sample_views]
            else:
                data["depth_layout_input"] = layout_scene_coord_maps[:T_in]
                data["depth_layout_target"] = layout_scene_coord_maps[T_in : self.num_sample_views]
            data["layout_sem_task_embeddings"] = layout_sem_task_embeddings
            data["layout_depth_task_embeddings"] = layout_depth_task_embeddings

        data["pose_out"] = target_Ts
        data["pose_in"] = cond_Ts
        data["plucker_rays_input"] = input_plucker_rays
        data["plucker_rays_target"] = target_plucker_rays
        data["rays_input"] = input_rays
        data["rays_target"] = target_rays
        data["color_task_embeddings"] = color_task_embeddings
        data["depth_task_embeddings"] = depth_task_embeddings
        if not self.use_scene_coord_map:
            data["depth_min"] = min_depth
            data["depth_max"] = max_depth
        else:
            data["depth_min"] = coord_min
            data["depth_max"] = coord_max
        data["scene_scale"] = curr_scene_scale
        data["input_indices"] = input_indices
        data["output_indices"] = target_indices
        if self.return_metric_data:
            data["pose_metric_input"] = metric_poses[:T_in]
            data["pose_metric_target"] = metric_poses[T_in : self.num_sample_views]
            data["depth_metric_input"] = metric_depths[:T_in]
            data["depth_metric_target"] = metric_depths[T_in : self.num_sample_views]
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                data["layout_depth_metric_input"] = render_depths[:T_in]
                data["layout_depth_metric_target"] = render_depths[T_in : self.num_sample_views]
        data["intrinsic"] = intrinsic_mat

        return data

    def get_spatiallm_item(self, sample_room_path: str | None = None) -> Dict:
        """
        Get item from [SpatialLM Testset](https://huggingface.co/datasets/manycore-research/SpatialLM-Testset/tree/main).
        Args:
            sample_room_path: str|None=None, room path

        Returns:
            Dict: spatialLM item
        """
        assert self.use_scene_coord_map, "get_spatiallm_item only supports use_scene_coord_map=True"

        room_uid = sample_room_path.split("/")[-1]
        rgb_dir = sample_room_path
        scene_layout_meta_file = os.path.join(sample_room_path, "room_layout.json")
        scene_cams_meta_file = os.path.join(sample_room_path, "cameras.json")
        scene_layout_mesh_file = os.path.join(sample_room_path, "layout_bbox.ply")
        valid_frames_file = os.path.join(sample_room_path, "valid_frames.txt")
        assert os.path.exists(scene_layout_meta_file), f"Layout metadata file {scene_layout_meta_file} does not exist."
        assert os.path.exists(scene_cams_meta_file), f"Camera metadata file {scene_cams_meta_file} does not exist."
        assert os.path.exists(scene_layout_mesh_file), f"Layout mesh file {scene_layout_mesh_file} does not exist."
        assert os.path.exists(valid_frames_file), f"Valid frames file {valid_frames_file} does not exist."

        # load camera metadata
        camera_data_dict = read_json(scene_cams_meta_file)
        # load layout metadata
        layout_data_dict = read_json(scene_layout_meta_file)

        # visualize the camera poses
        ori_width = camera_data_dict["width"]
        ori_height = camera_data_dict["height"]
        if ori_width != 1280 or ori_height != 720:
            raise NotImplementedError(f"Room {room_uid} has resolution {ori_width}x{ori_height}, Only support 1280x720 resolution")
        intrinsic_mat = torch.tensor(camera_data_dict["intrinsic"], dtype=torch.float32).reshape(3, 3)
        # update intrinsic matrix for the target image size, since crop only change the principal point,
        # we need to update the focal length according to the size ratio of the short side
        focal_scale = self.image_height / float(ori_height)
        intrinsic_mat[0, 0] *= focal_scale
        intrinsic_mat[1, 1] *= focal_scale
        intrinsic_mat[0, 2] = self.image_width / 2.0
        intrinsic_mat[1, 2] = self.image_height / 2.0
        # print(f"Cropped fx: {intrinsic_mat[0, 0]}, fy: {intrinsic_mat[1, 1]}, fov: {2 * np.arctan2(self.image_width / 2.0, intrinsic_mat[0, 0]) * 180 / np.pi}")

        # use the sequential sampling
        num_sample_views = self.num_sample_views
        T_in = self.T_in
        T_out = num_sample_views - T_in
        valid_frame_ids = readlines(valid_frames_file)[:360]
        step = len(valid_frame_ids) // num_sample_views
        sampled_camera_keys = valid_frame_ids[::step][:num_sample_views]
        cam_meta_data = camera_data_dict["cameras"]

        rgbs = []
        poses = []
        for cam_key in sampled_camera_keys:
            # load meta data
            c2w_pose = torch.from_numpy(np.array(cam_meta_data[cam_key]).reshape(4, 4).astype(np.float32)).float()
            poses.append(c2w_pose)

            # load rgb
            rgb_filepath = os.path.join(rgb_dir, cam_key + ".jpg")
            rgb = np.array(Image.open(rgb_filepath).convert("RGB")).astype(np.uint8)
            rgb: Float[Tensor, "1 3 H W"] = self._load_hypersim_rgb(rgb)  # (1, 3, H, W)

            # convert rgb to [-1, 1]
            normalized_rgb = rgb * 2.0 - 1.0
            rgbs.append(normalized_rgb)

        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.cat(rgbs, dim=0)

        metric_poses = poses.clone()
        
        obj_layout_dict, obj_mesh_list = parse_spatiallm_obj_bbox_from_meta(
            layout_data_dict, sample_room_path, return_mesh=True
        )
        render_layout_semantics = []
        render_layout_depths = []
        for i in range(self.num_sample_views):
            cam_in_obj_mask = compute_camera_inside_bbox(metric_poses[i], obj_layout_dict)[0]
            pose_device = self.cuda_device
            #  extract wall mesh
            wall_mesh = obj_mesh_list[-1]
            filtered_mesh_list = [obj_mesh_list[_id] for _id, _x in enumerate(cam_in_obj_mask) if not _x]

            # filtered_mesh_list.append(wall_mesh)
            concat_mesh = trimesh.util.concatenate(filtered_mesh_list)
            obj_mesh, face_color = trimesh_to_p3dmesh(concat_mesh)
            obj_mesh = obj_mesh.to(pose_device)
            face_color = face_color.to(pose_device)

            wall_mesh, wall_face_color = trimesh_to_p3dmesh(wall_mesh)
            wall_mesh = wall_mesh.to(pose_device)
            wall_face_color = wall_face_color.to(pose_device)
            w2c = metric_poses[i : i + 1].inverse()

            p3d_camera = cameras_from_opencv_projection(
                w2c[..., :3, :3],
                w2c[..., :3, 3],
                camera_matrix=intrinsic_mat[None,],
                image_size=torch.tensor([[self.image_height, self.image_width]]),
            ).to(pose_device)

            layout_rener_imgs = self.p3d_renderer(obj_mesh, cameras=p3d_camera, faces_color=face_color)
            wall_render_imgs = self.p3d_renderer(wall_mesh, cameras=p3d_camera, faces_color=wall_face_color)
            wall_render_sem_img = (wall_render_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)
            wall_render_dep_img = wall_render_imgs["depth"].cpu()[0].numpy().astype(np.float32)
            render_sem_img = (layout_rener_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)

            # fill the background area with wall_rendering
            bg_area = np.all(render_sem_img == 0, axis=-1)
            render_sem_img[bg_area] = wall_render_sem_img[bg_area]

            # fill the background area with wall_rendering
            render_depth_img = layout_rener_imgs["depth"].cpu()[0].numpy()
            render_depth_img[render_depth_img == -1.0] = wall_render_dep_img[render_depth_img == -1.0]
            # filter out the occluded pixels (when the farther object is occluded by the closer walls)
            occluded_area = np.logical_and(
                render_depth_img > (wall_render_dep_img + 0.5), wall_render_dep_img > 0
            ).squeeze()
            render_depth_img[occluded_area] = wall_render_dep_img[occluded_area]
            render_sem_img[occluded_area] = wall_render_sem_img[occluded_area]

            render_layout_semantics.append(TO_DIFFUSION_TENSOR(render_sem_img))
            render_layout_depths.append(torch.from_numpy(render_depth_img).permute(2, 0, 1).float())

            # # save semantic images
            # if True:
            #     merged_image = Image.new('RGB', (self.image_width, self.image_height * 2))
            #     merged_image.paste(Image.fromarray(render_sem_img.astype(np.uint8)), (0, 0))
            #     merged_image.paste(Image.fromarray(colorize_depth(render_depth_img[:,:,0]).astype(np.uint8)).convert('RGB'), (0, 1 * self.image_height))
            #     merged_image.save(f"{room_uid.replace('/', '_')}_{i}_pyt3d.png")

        render_semantics: Float[Tensor, "N 3 H W"] = torch.stack(render_layout_semantics, dim=0)
        render_depths: Float[Tensor, "N 1 H W"] = torch.stack(render_layout_depths, dim=0)

        # always choose the first view as the reference view
        relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
        # update poses to relative poses
        poses = relative_poses
        if self.use_scene_coord_map:
            layout_scene_coord_maps = unproject_depth(depth_map=render_depths, C2W=poses, K=intrinsic_mat)
            # normalize scene coord maps to [-1, 1]
            coord_max, coord_min = layout_scene_coord_maps.max(), layout_scene_coord_maps.min()
            # max_depth is scene range
            scene_range = torch.abs(coord_max - coord_min)
            layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (
                layout_scene_coord_maps - coord_min
            ) / scene_range * 2.0 - 1.0
            layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)
        else:
            # normalize depth to [-1, 1]
            normalized_render_depths, min_render_depth, max_render_depth = self.metric_depth_normalization(
                render_depths
            )

        if not self.use_metric_depth:
            # normalize poses to unit cube[-1,1] w.r.t current sample views
            curr_scale_mat, curr_scene_scale = self.calc_scale_mat(poses, depth_range=scene_range, offset_center=False)
            for pose_idx in range(poses.shape[0]):
                # scale pose_c2w
                subview_pose = curr_scale_mat @ poses[pose_idx]
                R_c2w = (subview_pose[:3, :3]).numpy()
                q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
                q_c2w = trimesh.transformations.unit_vector(q_c2w)
                R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
                subview_pose[:3, :3] = torch.from_numpy(R_c2w)
                poses[pose_idx] = subview_pose
        else:
            assert not self.use_scene_coord_map
            curr_scene_scale = 1

        fl_x, fl_y, cx, cy = (
            intrinsic_mat[0, 0],
            intrinsic_mat[1, 1],
            intrinsic_mat[0, 2],
            intrinsic_mat[1, 2],
        )
        directions = get_ray_directions(
            H=self.image_height,
            W=self.image_width,
            focal=[fl_x, fl_y],
            principal=[cx, cy],
        )
        canonical_ray_directions: Float[Tensor, "B H W 3"] = directions[None, :, :, :].repeat(num_sample_views, 1, 1, 1)
        rays_o, rays_d = get_rays(canonical_ray_directions, relative_poses, keepdim=True)
        rays_od = torch.cat([rays_o, rays_d], dim=-1)
        rays_od = rays_od.permute(0, 3, 1, 2)  # B, 6, H, W
        # plucker rays
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, H, W, 3
        plucker_rays = torch.cat([rays_dxo, rays_d], dim=-1)  # B, H, W, 6
        plucker_rays = plucker_rays.permute(0, 3, 1, 2)  # B, 6, H, W

        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:T_in]

        # atarget views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[T_in:num_sample_views]

        depth_class = torch.tensor([1, 0, 0, 0]).float()
        depth_task_embeddings = torch.stack([depth_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        color_class = torch.tensor([0, 1, 0, 0]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_normal:
            normal_class = torch.tensor([0, 0, 1, 0]).float()
            normal_task_embeddings = torch.stack([normal_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_semantic:
            semantic_class = torch.tensor([0, 0, 0, 1]).float()
            semantic_task_embeddings = torch.stack([semantic_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            layout_sem_class = torch.tensor([0, 0, 0, 10]).float()
            layout_sem_task_embeddings = torch.stack([layout_sem_class] * self.num_sample_views, dim=0)
            layout_depth_class = torch.tensor([0, 0, 0, 101]).float()
            layout_depth_task_embeddings = torch.stack([layout_depth_class] * self.num_sample_views, dim=0)

        cond_Ts: Float[Tensor, "N 4 4"] = poses[:T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[T_in : self.num_sample_views]

        input_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[:T_in]
        target_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[T_in : self.num_sample_views]

        input_rays: Float[Tensor, "N 6 H W"] = rays_od[:T_in]
        target_rays: Float[Tensor, "N 6 H W"] = rays_od[T_in : self.num_sample_views]

        # shuffled_indices = torch.randperm(self.num_sample_views)
        shuffled_indices = torch.arange(self.num_sample_views)
        input_indices = shuffled_indices[:T_in]
        target_indices = shuffled_indices[T_in:]

        data = {}
        data["dataset"] = "spatialgen"
        data["room_uid"] = room_uid
        data["image_input"] = input_images
        data["image_target"] = target_images

        if self.use_normal:
            data["normal_task_embeddings"] = normal_task_embeddings

        if self.use_semantic:
            data["semantic_task_embeddings"] = semantic_task_embeddings

        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            data["semantic_layout_input"] = render_semantics[:T_in]
            data["semantic_layout_target"] = render_semantics[T_in : self.num_sample_views]
            if not self.use_scene_coord_map:
                normalized_render_depths = normalized_render_depths.repeat(1, 3, 1, 1)
                data["depth_layout_input"] = normalized_render_depths[:T_in]
                data["depth_layout_target"] = normalized_render_depths[T_in : self.num_sample_views]
            else:
                data["depth_layout_input"] = layout_scene_coord_maps[:T_in]
                data["depth_layout_target"] = layout_scene_coord_maps[T_in : self.num_sample_views]
            data["layout_sem_task_embeddings"] = layout_sem_task_embeddings
            data["layout_depth_task_embeddings"] = layout_depth_task_embeddings

        data["pose_out"] = target_Ts
        data["pose_in"] = cond_Ts
        data["plucker_rays_input"] = input_plucker_rays
        data["plucker_rays_target"] = target_plucker_rays
        data["rays_input"] = input_rays
        data["rays_target"] = target_rays
        data["color_task_embeddings"] = color_task_embeddings
        data["depth_task_embeddings"] = depth_task_embeddings
        data["depth_min"] = coord_min
        data["depth_max"] = coord_max
        data["scene_scale"] = curr_scene_scale
        data["input_indices"] = input_indices
        data["output_indices"] = target_indices
        if self.return_metric_data:
            data["pose_metric_input"] = metric_poses[:T_in]
            data["pose_metric_target"] = metric_poses[T_in : self.num_sample_views]
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                data["layout_depth_metric_input"] = render_depths[:T_in]
                data["layout_depth_metric_target"] = render_depths[T_in : self.num_sample_views]
        data["intrinsic"] = intrinsic_mat

        return data

    # resize image size, return the image in [-1, 1]
    def resize_image(
        self, image: Float[Tensor, "1 3 H W"], image_type: str = "rgb", depth_scale: float = 1000.0
    ) -> Float[Tensor, "1 3 H W"]:
        interpolation = cv2.INTER_NEAREST if image_type in ["depth", "normal"] else cv2.INTER_LINEAR
        if image.shape[-2:] != (self.image_height, self.image_width):
            if image_type == "depth":
                pers_img = torch.from_numpy(
                    cv2.resize(
                        image[0].permute(1, 2, 0).cpu().numpy(),
                        (self.image_height, self.image_width),
                        interpolation=interpolation,
                    )
                )[None, :, :]
            else:
                pers_img = torch.from_numpy(
                    cv2.resize(
                        image[0].permute(1, 2, 0).cpu().numpy(),
                        (self.image_height, self.image_width),
                        interpolation=interpolation,
                    )
                ).permute(2, 0, 1)
        else:
            pers_img = image[0]

        if image_type in ["rgb", "semantic"]:
            # normalize rgb to [-1, 1]
            pers_img = pers_img.float() / 255.0
            pers_img = pers_img.clip(0.0, 1.0) * 2.0 - 1.0
        elif image_type in ["depth"]:
            pers_img = pers_img.float() / depth_scale
        elif image_type in ["normal"]:
            # convert normal to [-1, 1]
            normal_img = pers_img.permute(1, 2, 0).cpu().numpy()
            normal = np.clip((normal_img + 0.5) / 255.0, 0.0, 1.0) * 2 - 1
            normal = normal / (np.linalg.norm(normal, axis=2)[:, :, np.newaxis] + 1e-6)
            # save normal in camera space, flip to make +z upward
            pers_img = torch.from_numpy(normal).permute(2, 0, 1).float()  # [3, 256, 256]
        return pers_img

    def get_spatialgen_persp_item(
        self,
        sample_room_path: str | None = None,
        depth_scale: float = 1000.0,
    ) -> Dict:
        # sample_room_path = '/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_spiral_nonsquare/20241118/3FO4K5G1B29C/perspective/room_805'
        room_uid = "/".join(sample_room_path.split("/")[-4:])
        rgb_dir = os.path.join(sample_room_path, "rgb")
        depth_dir = os.path.join(sample_room_path, "depth")
        normal_dir = os.path.join(sample_room_path, "normal")
        semantic_dir = os.path.join(sample_room_path, "semantic")
        layout_semantic_dir = os.path.join(sample_room_path, "layout_semantic")
        layout_depth_dir = os.path.join(sample_room_path, "layout_depth")
        cam_meta_filepath = os.path.join(sample_room_path, "cameras.json")

        # load camera poses
        cameras_meta = read_json(cam_meta_filepath)
        total_valid_cam_keys = list(cameras_meta["cameras"].keys())

        # random the num of input views
        num_sample_views = self.num_sample_views
        T_in = self.T_in
        T_out = self.num_sample_views - T_in

        # use the sequential sampling
        start_index = np.random.randint(0, len(total_valid_cam_keys)) if not self.is_validation else 0
        valid_frame_ids = np.roll(total_valid_cam_keys, -start_index)
        if self.is_validation:
            valid_frame_ids = total_valid_cam_keys
        sample_keys = valid_frame_ids[:num_sample_views]

        # c2w poses, rgbs, depths, normals, semantics
        poses, rgbs = [], []
        depths, normals, semantics = [], [], []
        render_semantics, render_depths = [], []   # layout prior

        intrinsic_mat = torch.tensor(cameras_meta["intrinsic"], dtype=torch.float32).reshape(3, 3)
        ori_img_height, ori_img_width = cameras_meta["height"], cameras_meta["width"]
        # resize intrinsic matrix w.r.t the hight
        if self.image_height != ori_img_height or self.image_width != ori_img_width:
            intrinsic_mat[0, 0] *= self.image_height / ori_img_height
            intrinsic_mat[1, 1] *= self.image_height / ori_img_height
            intrinsic_mat[0, 2] = self.image_width / 2.0
            intrinsic_mat[1, 2] = self.image_height / 2.0
        focal_len = float(intrinsic_mat[0, 0])

        for idx, view_key in enumerate(sample_keys):

            # load c2w pose
            c2w_pose = torch.from_numpy(np.array(cameras_meta["cameras"][view_key]).reshape(4, 4)).float()
            # load perspective images
            frame_name = f"frame_{view_key}"
            rgb_image_filepath = os.path.join(rgb_dir, f"{frame_name}.jpg")
            depth_image_filepath = os.path.join(depth_dir, f"{frame_name}.png")
            normal_image_filepath = os.path.join(normal_dir, f"{frame_name}.jpg")
            semantic_image_filepath = os.path.join(semantic_dir, f"{frame_name}.jpg")

            rgb_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(rgb_image_filepath)
            depth_img: Float[Tensor, "1 1 H W"] = self._load_pil_image(depth_image_filepath)
            if self.use_normal:
                normal_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(normal_image_filepath)
            if self.use_semantic:
                semantic_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(semantic_image_filepath)
            if self.use_layout_prior:
                layout_semantic_image_filepath = os.path.join(layout_semantic_dir, f"{frame_name}.jpg")
                layout_depth_image_filepath = os.path.join(layout_depth_dir, f"{frame_name}.png")
                layout_semantic_img: Float[Tensor, "1 3 H W"] = self._load_pil_image(layout_semantic_image_filepath)
                layout_depth_img: Float[Tensor, "1 1 H W"] = self._load_pil_image(layout_depth_image_filepath)

            # scale rgb to [-1, 1]
            subview_rgbs = [self.resize_image(rgb_img, image_type="rgb")]
            subview_depths = [self.resize_image(depth_img, image_type="depth", depth_scale=depth_scale)]
            if self.use_normal:
                subview_normals = [self.resize_image(normal_img, image_type="normal")]
            if self.use_semantic:
                subview_semantics = [self.resize_image(semantic_img, image_type="semantic")]
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                subview_layout_semantics = [self.resize_image(layout_semantic_img, image_type="semantic")]
                subview_layout_depths = [
                    self.resize_image(layout_depth_img, image_type="depth", depth_scale=depth_scale)
                ]
                # complete depth_img with layout_depth
                subview_depths[0][subview_depths[0] < 1e-3] = subview_layout_depths[0][subview_depths[0] < 1e-3]
                # clamp depth to 12.5m
                subview_depths[0][subview_depths[0] > 12.5] = 12.5
                subview_layout_depths[0][subview_layout_depths[0] > 12.5] = 12.5
                render_semantics += subview_layout_semantics
                render_depths += subview_layout_depths

            subview_poses = [c2w_pose]

            poses += subview_poses
            rgbs += subview_rgbs
            depths += subview_depths
            if self.use_normal:
                normals += subview_normals
            if self.use_semantic:
                semantics += subview_semantics

        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.stack(rgbs, dim=0)
        depths: Float[Tensor, "N 1 H W"] = torch.stack(depths, dim=0)
        if self.use_normal:
            normals: Float[Tensor, "N 3 H W"] = torch.stack(normals, dim=0)
        if self.use_semantic:
            semantics: Float[Tensor, "N 3 H W"] = torch.stack(semantics, dim=0)
        if self.use_layout_prior:
            render_semantics: Float[Tensor, "N 3 H W"] = torch.stack(render_semantics, dim=0)
            render_depths: Float[Tensor, "N 1 H W"] = torch.stack(render_depths, dim=0)

        metric_poses = poses.clone()
        metric_depths = depths.clone()

        # always choose the first view as the reference view
        relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
        # update poses to relative poses
        poses = relative_poses

        # if self.use_scene_coord_map:
        # project depth to point cloud
        scene_coord_maps = unproject_depth(depth_map=depths, C2W=poses, K=intrinsic_mat)
        # normalize scene coord maps to [-1, 1]
        coord_max, coord_min = scene_coord_maps.max(), scene_coord_maps.min()
        # max_depth is scene range
        scene_range = torch.abs(coord_max - coord_min)
        scene_coord_maps: Float[Tensor, "N 3 H W"] = (scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
        scene_coord_maps = scene_coord_maps.clamp(-1.0, 1.0)

        if self.use_layout_prior:
            layout_scene_coord_maps = unproject_depth(depth_map=render_depths, C2W=poses, K=intrinsic_mat)
            # normalize scene coord maps to [-1, 1]
            layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (layout_scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
            layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)
        # else:
        #     # normalize depth to [-1, 1]
        #     if not self.use_metric_depth:
        #         normalized_depths, min_depth, max_depth = self.depth_scale_shift_normalization(depths)
        #         if self.use_layout_prior or self.use_layout_prior_from_p3d:
        #             assert self.use_metric_depth, "use_metric_depth should be True when use layout_depths!!!"
        #     else:
        #         normalized_depths, min_depth, max_depth = self.metric_depth_normalization(depths)
        #         if self.use_layout_prior or self.use_layout_prior_from_p3d:
        #             normalized_render_depths, min_render_depth, max_render_depth = self.metric_depth_normalization(
        #                 render_depths
        #             )

        # if not self.use_metric_depth:
        # normalize poses to unit cube[-1,1] w.r.t current sample views
        curr_scale_mat, curr_scene_scale = self.calc_scale_mat(poses, depth_range=scene_range, offset_center=False)
        for pose_idx in range(poses.shape[0]):
            # scale pose_c2w
            subview_pose = curr_scale_mat @ poses[pose_idx]
            R_c2w = (subview_pose[:3, :3]).numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            subview_pose[:3, :3] = torch.from_numpy(R_c2w)
            poses[pose_idx] = subview_pose
        # else:
        #     assert not self.use_scene_coord_map
        #     curr_scene_scale = 1

        canonical_ray_directions = self.prepare_ray_directions(num_sample_views=num_sample_views, focal_length=focal_len)
        rays_o, rays_d = get_rays(canonical_ray_directions, poses, keepdim=True)
        rays_od = torch.cat([rays_o, rays_d], dim=-1)
        rays_od = rays_od.permute(0, 3, 1, 2)  # N, 6, H, W

        # plucker rays
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # N, H, W, 3
        plucker_rays = torch.cat([rays_dxo, rays_d], dim=-1)  # N, H, W, 6
        plucker_rays = plucker_rays.permute(0, 3, 1, 2)  # N, 6, H, W

        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:T_in]

        # atarget views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[T_in : self.num_sample_views]

        # if not self.use_scene_coord_map:
        #     input_depths: Float[Tensor, "N 3 H W"] = normalized_depths[:T_in].repeat(1, 3, 1, 1)
        #     target_depths: Float[Tensor, "N 3 H W"] = normalized_depths[T_in : self.num_sample_views].repeat(1, 3, 1, 1)
        # else:
        input_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[:T_in]
        target_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[T_in : self.num_sample_views]

        if self.use_normal:
            input_normals: Float[Tensor, "N 3 H W"] = normals[:T_in]
            target_normals: Float[Tensor, "N 3 H W"] = normals[T_in : self.num_sample_views]
        if self.use_semantic:
            input_semantics: Float[Tensor, "N 3 H W"] = semantics[:T_in]
            target_semantics: Float[Tensor, "N 3 H W"] = semantics[T_in : self.num_sample_views]

        depth_class = torch.tensor([1, 0, 0, 0]).float()
        depth_task_embeddings = torch.stack([depth_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        color_class = torch.tensor([0, 1, 0, 0]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_normal:
            normal_class = torch.tensor([0, 0, 1, 0]).float()
            normal_task_embeddings = torch.stack([normal_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_semantic:
            semantic_class = torch.tensor([0, 0, 0, 1]).float()
            semantic_task_embeddings = torch.stack([semantic_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_layout_prior:
            layout_sem_class = torch.tensor([0, 0, 0, 10]).float()
            layout_sem_task_embeddings = torch.stack([layout_sem_class] * self.num_sample_views, dim=0)
            layout_depth_class = torch.tensor([0, 0, 0, 101]).float()
            layout_depth_task_embeddings = torch.stack([layout_depth_class] * self.num_sample_views, dim=0)

        cond_Ts: Float[Tensor, "N 4 4"] = poses[:T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[T_in : self.num_sample_views]

        input_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[:T_in]
        target_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[T_in : self.num_sample_views]

        input_rays: Float[Tensor, "N 6 H W"] = rays_od[:T_in]
        target_rays: Float[Tensor, "N 6 H W"] = rays_od[T_in : self.num_sample_views]

        shuffled_indices = torch.arange(self.num_sample_views)
        input_indices = shuffled_indices[:T_in]
        target_indices = shuffled_indices[T_in:]

        data = {}
        data["dataset"] = "spatialgen"
        data["room_uid"] = room_uid
        data["image_input"] = input_images
        data["image_target"] = target_images
        data["depth_input"] = input_depths
        data["depth_target"] = target_depths
        if self.use_normal:
            data["normal_input"] = input_normals
            data["normal_target"] = target_normals
            data["normal_task_embeddings"] = normal_task_embeddings

        if self.use_semantic:
            data["semantic_input"] = input_semantics
            data["semantic_target"] = target_semantics
            data["semantic_task_embeddings"] = semantic_task_embeddings

        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            data["semantic_layout_input"] = render_semantics[:T_in]
            data["semantic_layout_target"] = render_semantics[T_in : self.num_sample_views]
            if not self.use_scene_coord_map:
                normalized_render_depths = normalized_render_depths.repeat(1, 3, 1, 1)
                data["depth_layout_input"] = normalized_render_depths[:T_in]
                data["depth_layout_target"] = normalized_render_depths[T_in : self.num_sample_views]
            else:
                data["depth_layout_input"] = layout_scene_coord_maps[:T_in]
                data["depth_layout_target"] = layout_scene_coord_maps[T_in : self.num_sample_views]
            data["layout_sem_task_embeddings"] = layout_sem_task_embeddings
            data["layout_depth_task_embeddings"] = layout_depth_task_embeddings

        data["pose_out"] = target_Ts
        data["pose_in"] = cond_Ts
        data["plucker_rays_input"] = input_plucker_rays
        data["plucker_rays_target"] = target_plucker_rays
        data["rays_input"] = input_rays
        data["rays_target"] = target_rays
        data["color_task_embeddings"] = color_task_embeddings
        data["depth_task_embeddings"] = depth_task_embeddings
        # if not self.use_scene_coord_map:
        #     data["depth_min"] = min_depth
        #     data["depth_max"] = max_depth
        # else:
        data["depth_min"] = coord_min
        data["depth_max"] = coord_max
        data["scene_scale"] = curr_scene_scale
        data["input_indices"] = input_indices
        data["output_indices"] = target_indices
        if self.return_metric_data:
            data["pose_metric_input"] = metric_poses[:T_in]
            data["pose_metric_target"] = metric_poses[T_in : self.num_sample_views]
            data["depth_metric_input"] = metric_depths[:T_in]
            data["depth_metric_target"] = metric_depths[T_in : self.num_sample_views]
            data["layout_depth_metric_input"] = render_depths[:T_in]
            data["layout_depth_metric_target"] = render_depths[T_in : self.num_sample_views]
        data["intrinsic"] = intrinsic_mat

        return data

    def _load_hypersim_rgb(self, image: np.ndarray):
        image_pil = Image.fromarray(image)
        image = self.read_image_file(
            image_pil,
            height=self.image_height,
            width=self.image_width,
            resampling_mode=Image.BILINEAR,
            disable_warning=False,
            target_aspect_ratio=self.image_width / self.image_height,
        )
        image = image.unsqueeze(0)
        return image

    def _load_hypersim_depth(self, depth_map: np.ndarray, depth_scale: float = 1.0):
        depth_pil = Image.fromarray(depth_map)
        depth = self.read_image_file(
            depth_pil,
            height=self.image_height,
            width=self.image_width,
            resampling_mode=Image.NEAREST,
            disable_warning=False,
            target_aspect_ratio=self.image_width / self.image_height,
        )

        depth = depth[None, :, :] / depth_scale
        return depth

    def get_hypersim_item(self, sample_room_path: str, depth_scale: float = 1.0) -> Dict:
        ori_height, ori_width = 768, 1024
        room_uid = sample_room_path.split("/")[-1]

        cam_keys = [cam[:-4] for cam in os.listdir(sample_room_path) if cam.endswith(".npz")]
        cam_keys = sorted(cam_keys, key=lambda x: int(x.split(".")[-1]))

        num_sample_views = self.num_sample_views
        # random the num of input views
        T_in = self.T_in
        T_out = self.num_sample_views - T_in

        sampled_camera_keys = []
        # use the sequential sampling
        # if self.cam_sample_type == "sequential":
        # start_index = np.random.randint(0, len(cam_keys))
        # valid_frame_ids = np.roll(cam_keys, -start_index)
        step = len(cam_keys) // num_sample_views
        if self.is_validation:
            step = 1
        sampled_camera_keys = cam_keys[3::step][:num_sample_views]

        rgbs, depths, semantics, layout_semantics, layout_depths = [], [], [], [], []
        controlnet_rgbs = []
        intrinsics, poses = [], []
        for cam_key in sampled_camera_keys:
            # load meta data
            cam_meta_filepath = os.path.join(sample_room_path, cam_key + ".npz")
            meta_data = np.load(cam_meta_filepath)
            c2w_gl = torch.from_numpy(meta_data["extrin"]).reshape(4, 4).float()
            c2w_pose = opengl_to_opencv(c2w_gl)
            intrin = torch.from_numpy(meta_data["intrin"]).float()[:3, :3]

            poses.append(c2w_pose)
            intrinsics.append(intrin)

            # load rgb, depth, semantic
            rgb = meta_data["frame_rgb"].astype(np.uint8)
            rgb: Float[Tensor, "1 3 H W"] = self._load_hypersim_rgb(rgb)  # (1, 3, H, W)

            depth = meta_data["frame_depth"].astype(np.float32)
            # depth = convert_distance_to_z(distance, image_height=ori_height, image_width=ori_width, focal_length=float(intrin[0,0]))
            depth: Float[Tensor, "1 1 H W"] = self._load_hypersim_depth(depth, depth_scale=depth_scale)  # (1, 1, H, W)

            semantic = meta_data["frame_semantic"].astype(np.uint8)
            semantic: Float[Tensor, "1 3 H W"] = self._load_hypersim_rgb(semantic)

            # load layout_semantic, layout_depth image
            layout_semantic_img_filepath = os.path.join(sample_room_path, cam_key + ".png")
            layout_sem_img_pil = (
                Image.open(layout_semantic_img_filepath).convert("RGB").crop((0, ori_height, ori_width, ori_height * 2))
            )
            layout_semantic = self._load_hypersim_rgb(np.array(layout_sem_img_pil))  # (1, 3, H, W)
            layout_depth = meta_data["depths"].astype(np.float32)
            layout_depth: Float[Tensor, "1 1 H W"] = self._load_hypersim_depth(
                layout_depth, depth_scale=depth_scale
            )  # (1, 1, H, W)
            # complete depth_img with layout_depth
            depth[depth < 1e-3] = layout_depth[depth < 1e-3]

            if self.controlnet_output_dir is not None:
                controlnet_output_dir = os.path.join(self.controlnet_output_dir, room_uid, "controlnet_output")
                if self.is_validation and os.path.exists(controlnet_output_dir):
                    # load controlnet output
                    controlnet_rgb_filepath = os.path.join(controlnet_output_dir, cam_key + ".png")
                    contrlnet_rgb_img_pil = (
                        Image.open(controlnet_rgb_filepath).convert("RGB").crop((0, ori_height, ori_width, ori_height * 2))
                    )
                    controlnet_rgb = self._load_hypersim_rgb(np.array(contrlnet_rgb_img_pil))  # (1, 3, H, W)
                    controlnet_rgbs.append(controlnet_rgb * 2.0 - 1.0)

            # convert rgb to [-1, 1]
            normalized_rgb = rgb * 2.0 - 1.0
            rgbs.append(normalized_rgb)
            depths.append(depth)
            normalized_sem = semantic * 2.0 - 1.0
            semantics.append(normalized_sem)
            normalized_layout_sem = layout_semantic * 2.0 - 1.0
            layout_semantics.append(normalized_layout_sem)
            layout_depths.append(layout_depth)

        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.cat(rgbs, dim=0)
        depths: Float[Tensor, "N 1 H W"] = torch.cat(depths, dim=0)
        semantics: Float[Tensor, "N 3 H W"] = torch.cat(semantics, dim=0)
        layout_semantics: Float[Tensor, "N 3 H W"] = torch.cat(layout_semantics, dim=0)
        layout_depths: Float[Tensor, "N 1 H W"] = torch.cat(layout_depths, dim=0)
        if len(controlnet_rgbs) > 0:
            controlnet_rgbs: Float[Tensor, "N 3 H W"] = torch.cat(controlnet_rgbs, dim=0)
        else:
            controlnet_rgbs = None

        # update intrinsic matrix for the target image size, since crop only change the principal point,
        # we need to update the focal length according to the size ratio of the short side
        intrinsic_mat = intrinsics[0]
        focal_scale = self.image_height / float(ori_height)
        intrinsic_mat[0, 0] *= focal_scale
        intrinsic_mat[1, 1] *= focal_scale
        intrinsic_mat[0, 2] = self.image_width / 2.0
        intrinsic_mat[1, 2] = self.image_height / 2.0

        metric_poses = poses.clone()
        metric_depths = depths.clone()

        # always choose the first view as the reference view
        relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
        # update poses to relative poses
        poses = relative_poses

        if self.use_scene_coord_map:
            # project depth to point cloud
            scene_coord_maps = unproject_depth(depth_map=depths, C2W=poses, K=intrinsic_mat)
            # normalize scene coord maps to [-1, 1]
            coord_max, coord_min = scene_coord_maps.max(), scene_coord_maps.min()
            # max_depth is scene range
            scene_range = torch.abs(coord_max - coord_min)
            scene_coord_maps: Float[Tensor, "N 3 H W"] = (scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
            scene_coord_maps = scene_coord_maps.clamp(-1.0, 1.0)

            # if self.use_layout_prior or self.use_layout_prior_from_p3d:
            layout_scene_coord_maps = unproject_depth(depth_map=layout_depths, C2W=poses, K=intrinsic_mat)
            # normalize scene coord maps to [-1, 1]
            layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (
                layout_scene_coord_maps - coord_min
            ) / scene_range * 2.0 - 1.0
            layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)
        else:
            # normalize depth to [-1, 1]
            if not self.use_metric_depth:
                normalized_depths, min_depth, max_depth = self.depth_scale_shift_normalization(depths)
                # if self.use_layout_prior or self.use_layout_prior_from_p3d:
                # assert self.use_metric_depth, "use_metric_depth should be True when use layout_depths!!!"
                normalized_layout_depths, min_layout_depth, max_layout_depth = self.depth_scale_shift_normalization(
                    layout_depths
                )
            else:
                normalized_depths, min_depth, max_depth = self.metric_depth_normalization(depths)
                # if self.use_layout_prior or self.use_layout_prior_from_p3d:
                normalized_layout_depths, min_layout_depth, max_layout_depth = self.metric_depth_normalization(
                    layout_depths
                )

        if not self.use_metric_depth:
            # normalize poses to unit cube[-1,1] w.r.t current sample views
            curr_scale_mat, curr_scene_scale = self.calc_scale_mat(
                poses, depth_range=scene_range if self.use_scene_coord_map else max_depth, offset_center=False
            )
            for pose_idx in range(poses.shape[0]):
                # scale pose_c2w
                subview_pose = curr_scale_mat @ poses[pose_idx]
                R_c2w = (subview_pose[:3, :3]).numpy()
                q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
                q_c2w = trimesh.transformations.unit_vector(q_c2w)
                R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
                subview_pose[:3, :3] = torch.from_numpy(R_c2w)
                poses[pose_idx] = subview_pose
        else:
            assert not self.use_scene_coord_map, "use_scene_coord_map should be False when use_metric_depth is True!!!"
            curr_scene_scale = 1

        # # calculate rays
        # relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses

        fl_x, fl_y, cx, cy = (
            intrinsic_mat[0, 0],
            intrinsic_mat[1, 1],
            intrinsic_mat[0, 2],
            intrinsic_mat[1, 2],
        )
        directions = get_ray_directions(
            H=self.image_height,
            W=self.image_width,
            focal=[fl_x, fl_y],
            principal=[cx, cy],
        )
        canonical_ray_directions: Float[Tensor, "B H W 3"] = directions[None, :, :, :].repeat(
            num_sample_views, 1, 1, 1
        )
        rays_o, rays_d = get_rays(canonical_ray_directions, relative_poses, keepdim=True)
        rays_od = torch.cat([rays_o, rays_d], dim=-1)
        rays_od = rays_od.permute(0, 3, 1, 2)  # B, 6, H, W
        # plucker rays
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, H, W, 3
        plucker_rays = torch.cat([rays_dxo, rays_d], dim=-1)  # B, H, W, 6
        plucker_rays = plucker_rays.permute(0, 3, 1, 2)  # B, 6, H, W
        # # update poses to relative poses
        # poses = relative_poses

        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:T_in]

        # atarget views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[T_in : self.num_sample_views]

        if not self.use_scene_coord_map:
            input_depths: Float[Tensor, "N 3 H W"] = normalized_depths[:T_in].repeat(1, 3, 1, 1)
            target_depths: Float[Tensor, "N 3 H W"] = normalized_depths[T_in : self.num_sample_views].repeat(1, 3, 1, 1)
        else:
            input_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[:T_in]
            target_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[T_in : self.num_sample_views]

        # input_depths: Float[Tensor, "N 3 H W"] = normalized_depths[:T_in].repeat(1, 3, 1, 1)
        # target_depths: Float[Tensor, "N 3 H W"] = normalized_depths[T_in:self.num_sample_views].repeat(1, 3, 1, 1)

        input_semantics: Float[Tensor, "N 3 H W"] = semantics[:T_in]
        target_semantics: Float[Tensor, "N 3 H W"] = semantics[T_in : self.num_sample_views]

        depth_class = torch.tensor([1, 0, 0, 0]).float()
        depth_task_embeddings = torch.stack([depth_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        color_class = torch.tensor([0, 1, 0, 0]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        semantic_class = torch.tensor([0, 0, 0, 1]).float()
        semantic_task_embeddings = torch.stack([semantic_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        # if self.use_layout_prior and self.use_layout_prior_from_p3d:
        layout_sem_class = torch.tensor([0, 0, 0, 10]).float()
        layout_sem_task_embeddings = torch.stack([layout_sem_class] * self.num_sample_views, dim=0)
        layout_depth_class = torch.tensor([0, 0, 0, 101]).float()
        layout_depth_task_embeddings = torch.stack([layout_depth_class] * self.num_sample_views, dim=0)

        cond_Ts: Float[Tensor, "N 4 4"] = poses[:T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[T_in : self.num_sample_views]

        input_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[:T_in]
        target_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[T_in : self.num_sample_views]

        input_rays: Float[Tensor, "N 6 H W"] = rays_od[:T_in]
        target_rays: Float[Tensor, "N 6 H W"] = rays_od[T_in : self.num_sample_views]

        shuffled_indices = torch.arange(num_sample_views)
        input_indices = shuffled_indices[:T_in]
        target_indices = shuffled_indices[T_in:]

        data = {}
        data["dataset"] = "hypersim"
        data["room_uid"] = room_uid
        data["image_input"] = input_images
        data["image_target"] = target_images
        data["depth_input"] = input_depths
        data["depth_target"] = target_depths
        data["semantic_input"] = input_semantics
        data["semantic_target"] = target_semantics
        data["semantic_task_embeddings"] = semantic_task_embeddings
        # if self.use_layout_prior or self.use_layout_prior_from_p3d:
        data["semantic_layout_input"] = layout_semantics[:T_in]
        data["semantic_layout_target"] = layout_semantics[T_in : self.num_sample_views]
        if not self.use_scene_coord_map:
            normalized_layout_depths = normalized_layout_depths.repeat(1, 3, 1, 1)
            data["depth_layout_input"] = normalized_layout_depths[:T_in]
            data["depth_layout_target"] = normalized_layout_depths[T_in : self.num_sample_views]
        else:
            data["depth_layout_input"] = layout_scene_coord_maps[:T_in]
            data["depth_layout_target"] = layout_scene_coord_maps[T_in : self.num_sample_views]
        data["layout_sem_task_embeddings"] = layout_sem_task_embeddings
        data["layout_depth_task_embeddings"] = layout_depth_task_embeddings

        data["pose_out"] = target_Ts
        data["pose_in"] = cond_Ts
        data["plucker_rays_input"] = input_plucker_rays
        data["plucker_rays_target"] = target_plucker_rays
        data["rays_input"] = input_rays
        data["rays_target"] = target_rays
        data["color_task_embeddings"] = color_task_embeddings
        data["depth_task_embeddings"] = depth_task_embeddings
        if not self.use_scene_coord_map:
            data["depth_min"] = min_depth
            data["depth_max"] = max_depth
        else:
            data["depth_min"] = coord_min
            data["depth_max"] = coord_max
        data["scene_scale"] = curr_scene_scale
        data["input_indices"] = input_indices
        data["output_indices"] = target_indices
        if self.return_metric_data:
            data["pose_metric_input"] = metric_poses[:T_in]
            data["pose_metric_target"] = metric_poses[T_in : self.num_sample_views]
            data["depth_metric_input"] = metric_depths[:T_in]
            data["depth_metric_target"] = metric_depths[T_in : self.num_sample_views]
            data["layout_depth_metric_input"] = layout_depths[:T_in]
            data["layout_depth_metric_target"] = layout_depths[T_in : self.num_sample_views]
        data["intrinsic"] = intrinsic_mat
        if self.use_add_supervision_view:
            data["image_supervised"] = super_images
            data["depth_supervised"] = super_depths
            data["semantic_supervised"] = super_semantics
            data["pose_supervised"] = super_Ts
            data["plucker_rays_supervised"] = super_plucker_rays
            data["rays_supervised"] = super_rays
        if self.is_validation and controlnet_rgbs is not None:
            data["controlnet_image_input"] = controlnet_rgbs[:T_in]
            data["controlnet_image_target"] = controlnet_rgbs[T_in : self.num_sample_views]
        return data

    def get_structured3d_layout(self, sample_room_path: str, return_mesh=False, noceil_mesh=False) -> Dict:
        """parse 3D object layout in the sampled room

        Args:
            sample_room_path (str): room data path

        Returns:
            Dict: dict of object layout
        """
        scene_layout_meta_file = os.path.join(sample_room_path, "room_layout.json")
        assert os.path.exists(scene_layout_meta_file), f"Layout metadata file {scene_layout_meta_file} does not exist."

        # load layout metadata
        layout_data_dict = read_json(scene_layout_meta_file)

        return parse_spatiallm_obj_bbox_from_meta(layout_data_dict, sample_room_path, return_mesh=return_mesh)
    
    def get_structured3d_item(self, sample_room_path: str, depth_scale: float = 1000.0) -> Dict:
        """
        Get item from [Ctrl-Room Testset](https://arxiv.org/pdf/2310.03602).
        Args:
            sample_room_path: str|None=None, room path

        Returns:
            Dict: Structured3D item
        """
        room_uid = sample_room_path.split("/")[-1]
        assert self.use_scene_coord_map, "get_structured3d_item only supports use_scene_coord_map=True"

        room_uid = sample_room_path.split("/")[-1]
        rgb_dir = os.path.join(sample_room_path, "rgb")
        depth_dir = os.path.join(sample_room_path, "depth")
        scene_layout_meta_file = os.path.join(sample_room_path, "room_layout.json")
        scene_cams_meta_file = os.path.join(sample_room_path, "cameras.json")
        assert os.path.exists(scene_layout_meta_file), f"Layout metadata file {scene_layout_meta_file} does not exist."
        assert os.path.exists(scene_cams_meta_file), f"Camera metadata file {scene_cams_meta_file} does not exist."

        # load camera metadata
        camera_data_dict = read_json(scene_cams_meta_file)
        # load layout metadata
        layout_data_dict = read_json(scene_layout_meta_file)

        # visualize the camera poses
        cam_meta_data = camera_data_dict["cameras"]
        ori_width = camera_data_dict["width"]
        ori_height = camera_data_dict["height"]
        if ori_width != 1024 or ori_height != 512:
            raise NotImplementedError(f"Room {room_uid} has resolution {ori_width}x{ori_height}, Only support 1024x512 resolution")

        # TODO: random the num of input views
        T_in = self.T_in
        T_out = self.num_sample_views - T_in
        num_sample_views = self.num_sample_views

        # sample different camera trajectory
        # view_sampler = ViewSampler(
        #     room_metadata=layout_data_dict,
        #     num_sample_views=self.num_sample_views,
        #     yaw_interval_thresh=60.0,
        #     distance_interval_thresh=1.0,
        #     trajectory_type=self.cam_sample_type,
        #     is_validation=self.is_validation,
        # )
        # sample_dict = view_sampler.sample(
        #     valid_frames=total_pano_view_keys,
        #     cameras_dict=cameras_meta,
        #     b_make_sampled_views_consecutive=self.sample_consecutive_views,
        #     room_uid=room_uid,
        # )
        # use panoramic trajectory, TODO: use different trajectory
        sample_keys = np.arange(self.num_sample_views)  # sampled view keys
        # sample_yaws = np.arange(0, 360, 360.0 / self.num_sample_views).astype(np.float32)  # sampled yaw angles, [0, 360)
        sample_yaws = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] * (self.num_sample_views // 8), dtype=np.float32) # sampled yaw angles, [0, 360)
        sample_rolls = [float(0.0)] * self.num_sample_views  # sampled roll angles
        sample_pitches = [float(0.0)] * self.num_sample_views  # sampled pitch angles
        # c2w poses, rgbs, background colors
        poses, rgbs = [], []

        # TODO: make fov_x random
        random_fov_x = 90.
        focal_len = 0.5 * self.image_width / math.tan(random_fov_x * math.pi / 360.0)
        normalized_focal_len = 2.0 * focal_len / self.image_width
        intrinsic_mat = torch.tensor(
            [
                [focal_len, 0, self.image_width / 2.0],
                [0, focal_len, self.image_height / 2.0],
                [0, 0, 1],
            ]
        ).float()
        
        # use the sequential sampling
        cam0_c2w_pose = torch.from_numpy(np.array(cam_meta_data["0"]).reshape(4, 4).astype(np.float32)).float()
        c2w_positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [-0.5, -0.5, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.1, 0.1, 0.0],
            
        ], dtype=torch.float32)

        rgbs, depths = [], []
        poses = []
        controlnet_rgbs = []
        for idx, view_key in enumerate(sample_keys):
            subview_yaw_degree = sample_yaws[idx]

            # load c2w pose of panorama image
            c2w_pose = cam0_c2w_pose
            c2w_pose[:3, 3] = c2w_positions[idx]
                
            # load single equirectangular image
            rgb_pano_image_filepath = os.path.join(rgb_dir, "frame_0.png")
            depth_pano_image_filepath = os.path.join(depth_dir, "frame_0.png")

            equi_rgb: Float[Tensor, "1 3 H W"] = self._load_pil_image(rgb_pano_image_filepath)
            equi_depth: Float[Tensor, "1 1 H W"] = self._load_pil_image(depth_pano_image_filepath)
            # subview degrees
            subview_yaw_degrees = [subview_yaw_degree]  # [0, 360]
            subview_pitch_degrees = [sample_pitches[idx]]
            subview_roll_degress = [sample_rolls[idx]]

            rgb_subview_lst = self.cubemap_to_persepctive_images(
                equi_img=equi_rgb,
                subview_fov_x=random_fov_x,
                subview_yaws_lst=subview_yaw_degrees,
                subview_pitchs_lst=subview_pitch_degrees,
                subview_rolls_lst=subview_roll_degress,
                subview_img_resolution=self.image_width,
                image_type="rgb",
                disable_img_grad_check=True,  # check the gradient of the image
            )
            selected_subview_yaws = [rgb.yaw_angle for rgb in rgb_subview_lst]
            selected_subview_pitchs = [rgb.pitch_angle for rgb in rgb_subview_lst]
            selected_subview_rolls = [rgb.roll_angle for rgb in rgb_subview_lst]

            # assume len(selected_subview_yaws) is always 1
            depth_subview_lst = self.cubemap_to_persepctive_images(
                equi_img=equi_depth,
                subview_fov_x=random_fov_x,
                subview_yaws_lst=selected_subview_yaws,
                subview_pitchs_lst=selected_subview_pitchs,
                subview_rolls_lst=selected_subview_rolls,
                subview_img_resolution=self.image_width,
                depth_scale=depth_scale,
                image_type="depth",
                normalized_focal_length=normalized_focal_len,
            )

            # scale rgb to [-1, 1]
            subview_rgbs = [rgb.image * 2.0 - 1.0 for rgb in rgb_subview_lst]
            subview_depths = [depth.image for depth in depth_subview_lst]
            subview_poses = [
                self.calc_subview_pose(c2w_pose, torch.eye(4).float(), -rgb.yaw_angle, rgb.pitch_angle, rgb.roll_angle)
                for rgb in rgb_subview_lst
            ]

            if self.controlnet_output_dir is not None:
                controlnet_output_dir = os.path.join(self.controlnet_output_dir, room_uid)
                if self.is_validation and os.path.exists(controlnet_output_dir):
                    # load controlnet output
                    controlnet_rgb_filepath = os.path.join(controlnet_output_dir, f"frame_{view_key}.png")
                    contrlnet_rgb_img_pil = Image.open(controlnet_rgb_filepath).convert("RGB")
                    controlnet_rgb = self._load_hypersim_rgb(np.array(contrlnet_rgb_img_pil))  # (1, 3, H, W)
                    controlnet_rgbs.append(controlnet_rgb * 2.0 - 1.0)
                    
            poses += subview_poses
            rgbs += subview_rgbs
            depths += subview_depths

        poses: Float[Tensor, "N 4 4"] = torch.stack(poses, dim=0)
        rgbs: Float[Tensor, "N 3 H W"] = torch.cat(rgbs, dim=0)
        depths: Float[Tensor, "N 1 H W"] = torch.cat(depths, dim=0)
        if len(controlnet_rgbs) > 0:
            controlnet_rgbs: Float[Tensor, "N 3 H W"] = torch.cat(controlnet_rgbs, dim=0)
        else:
            controlnet_rgbs = None
            
        metric_poses = poses.clone()

        obj_layout_dict, obj_mesh_list = parse_spatiallm_obj_bbox_from_meta(
            layout_data_dict, sample_room_path, return_mesh=True
        )
        render_layout_semantics = []
        render_layout_depths = []
        for i in range(self.num_sample_views):
            cam_in_obj_mask = compute_camera_inside_bbox(metric_poses[i], obj_layout_dict)[0]
            pose_device = self.cuda_device
            #  extract wall mesh
            wall_mesh = obj_mesh_list[-1]
            filtered_mesh_list = [obj_mesh_list[_id] for _id, _x in enumerate(cam_in_obj_mask) if not _x]

            # filtered_mesh_list.append(wall_mesh)
            concat_mesh = trimesh.util.concatenate(filtered_mesh_list)
            obj_mesh, face_color = trimesh_to_p3dmesh(concat_mesh)
            obj_mesh = obj_mesh.to(pose_device)
            face_color = face_color.to(pose_device)

            wall_mesh, wall_face_color = trimesh_to_p3dmesh(wall_mesh)
            wall_mesh = wall_mesh.to(pose_device)
            wall_face_color = wall_face_color.to(pose_device)
            w2c = metric_poses[i : i + 1].inverse()

            p3d_camera = cameras_from_opencv_projection(
                w2c[..., :3, :3],
                w2c[..., :3, 3],
                camera_matrix=intrinsic_mat[None,],
                image_size=torch.tensor([[self.image_height, self.image_width]]),
            ).to(pose_device)

            layout_rener_imgs = self.p3d_renderer(obj_mesh, cameras=p3d_camera, faces_color=face_color)
            wall_render_imgs = self.p3d_renderer(wall_mesh, cameras=p3d_camera, faces_color=wall_face_color)
            wall_render_sem_img = (wall_render_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)
            wall_render_dep_img = wall_render_imgs["depth"].cpu()[0].numpy().astype(np.float32)
            render_sem_img = (layout_rener_imgs["render_segment"].cpu()[0].numpy() * 255).astype(np.uint8)

            # fill the background area with wall_rendering
            bg_area = np.all(render_sem_img == 0, axis=-1)
            render_sem_img[bg_area] = wall_render_sem_img[bg_area]

            # fill the background area with wall_rendering
            render_depth_img = layout_rener_imgs["depth"].cpu()[0].numpy()
            render_depth_img[render_depth_img == -1.0] = wall_render_dep_img[render_depth_img == -1.0]
            # filter out the occluded pixels (when the farther object is occluded by the closer walls)
            occluded_area = np.logical_and(
                render_depth_img > (wall_render_dep_img + 0.5), wall_render_dep_img > 0
            ).squeeze()
            render_depth_img[occluded_area] = wall_render_dep_img[occluded_area]
            render_sem_img[occluded_area] = wall_render_sem_img[occluded_area]

            render_layout_semantics.append(TO_DIFFUSION_TENSOR(render_sem_img))
            render_layout_depths.append(torch.from_numpy(render_depth_img).permute(2, 0, 1).float())

            # save semantic images
            # if True:
                # merged_image = Image.new('RGB', (self.image_width, self.image_height * 3))
                # merged_image.paste(Image.fromarray(render_sem_img.astype(np.uint8)), (0, 0))
                # merged_image.paste(Image.fromarray(colorize_depth(render_depth_img[:,:,0]).astype(np.uint8)).convert('RGB'), (0, 1 * self.image_height))
                # merged_image.paste(Image.fromarray(((rgbs[i] * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)), (0, 2 * self.image_height))
                # merged_image.save(f"{room_uid.replace('/', '_')}_{i}_pyt3d.png")
                # layout_sem_img_dir = os.path.join(sample_room_path, "layout_semantic")
                # os.makedirs(layout_sem_img_dir, exist_ok=True)
                # layout_sem_img_path = os.path.join(layout_sem_img_dir, f"frame_{i}.jpg")
                # Image.fromarray(render_sem_img.astype(np.uint8)).save(layout_sem_img_path)
                # layout_depth_img_dir = os.path.join(sample_room_path, "layout_depth")
                # os.makedirs(layout_depth_img_dir, exist_ok=True)
                # layout_depth_img_path = os.path.join(layout_depth_img_dir, f"frame_{i}.png")
                # vis_layout_depth = (render_depth_img[:,:,0] * 1000).astype(np.uint16)
                # Image.fromarray(vis_layout_depth).save(layout_depth_img_path)

        render_semantics: Float[Tensor, "N 3 H W"] = torch.stack(render_layout_semantics, dim=0)
        render_depths: Float[Tensor, "N 1 H W"] = torch.stack(render_layout_depths, dim=0)

        # always choose the first view as the reference view
        relative_poses = torch.inverse(poses[0:1]).repeat(poses.shape[0], 1, 1) @ poses
        # update poses to relative poses
        poses = relative_poses
        if self.use_scene_coord_map:
            # project depth to point cloud
            scene_coord_maps = unproject_depth(depth_map=depths, C2W=poses, K=intrinsic_mat)
            # normalize scene coord maps to [-1, 1]
            coord_max, coord_min = scene_coord_maps.max(), scene_coord_maps.min()
            # max_depth is scene range
            scene_range = torch.abs(coord_max - coord_min)
            scene_coord_maps: Float[Tensor, "N 3 H W"] = (scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
            scene_coord_maps = scene_coord_maps.clamp(-1.0, 1.0)
            
            layout_scene_coord_maps = unproject_depth(depth_map=render_depths, C2W=poses, K=intrinsic_mat)
            # # normalize scene coord maps to [-1, 1]
            # coord_max, coord_min = layout_scene_coord_maps.max(), layout_scene_coord_maps.min()
            # # max_depth is scene range
            # scene_range = torch.abs(coord_max - coord_min)
            layout_scene_coord_maps: Float[Tensor, "N 3 H W"] = (layout_scene_coord_maps - coord_min) / scene_range * 2.0 - 1.0
            layout_scene_coord_maps = layout_scene_coord_maps.clamp(-1.0, 1.0)

        if not self.use_metric_depth:
            # normalize poses to unit cube[-1,1] w.r.t current sample views
            curr_scale_mat, curr_scene_scale = self.calc_scale_mat(poses, depth_range=scene_range, offset_center=False)
            for pose_idx in range(poses.shape[0]):
                # scale pose_c2w
                subview_pose = curr_scale_mat @ poses[pose_idx]
                R_c2w = (subview_pose[:3, :3]).numpy()
                q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
                q_c2w = trimesh.transformations.unit_vector(q_c2w)
                R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
                subview_pose[:3, :3] = torch.from_numpy(R_c2w)
                poses[pose_idx] = subview_pose

        fl_x, fl_y, cx, cy = (
            intrinsic_mat[0, 0],
            intrinsic_mat[1, 1],
            intrinsic_mat[0, 2],
            intrinsic_mat[1, 2],
        )
        directions = get_ray_directions(
            H=self.image_height,
            W=self.image_width,
            focal=[fl_x, fl_y],
            principal=[cx, cy],
        )
        canonical_ray_directions: Float[Tensor, "B H W 3"] = directions[None, :, :, :].repeat(num_sample_views, 1, 1, 1)
        rays_o, rays_d = get_rays(canonical_ray_directions, relative_poses, keepdim=True)
        rays_od = torch.cat([rays_o, rays_d], dim=-1)
        rays_od = rays_od.permute(0, 3, 1, 2)  # B, 6, H, W
        # plucker rays
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, H, W, 3
        plucker_rays = torch.cat([rays_dxo, rays_d], dim=-1)  # B, H, W, 6
        plucker_rays = plucker_rays.permute(0, 3, 1, 2)  # B, 6, H, W

        # source views
        input_images: Float[Tensor, "N 3 H W"] = rgbs[:T_in]

        # atarget views
        target_images: Float[Tensor, "N 3 H W"] = rgbs[T_in:num_sample_views]

        input_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[:T_in]
        target_depths: Float[Tensor, "N 3 H W"] = scene_coord_maps[T_in : self.num_sample_views]
        
        depth_class = torch.tensor([1, 0, 0, 0]).float()
        depth_task_embeddings = torch.stack([depth_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        color_class = torch.tensor([0, 1, 0, 0]).float()
        color_task_embeddings = torch.stack([color_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_normal:
            normal_class = torch.tensor([0, 0, 1, 0]).float()
            normal_task_embeddings = torch.stack([normal_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_semantic:
            semantic_class = torch.tensor([0, 0, 0, 1]).float()
            semantic_task_embeddings = torch.stack([semantic_class] * self.num_sample_views, dim=0)  # (T_out+T_in, 4)
        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            layout_sem_class = torch.tensor([0, 0, 0, 10]).float()
            layout_sem_task_embeddings = torch.stack([layout_sem_class] * self.num_sample_views, dim=0)
            layout_depth_class = torch.tensor([0, 0, 0, 101]).float()
            layout_depth_task_embeddings = torch.stack([layout_depth_class] * self.num_sample_views, dim=0)

        cond_Ts: Float[Tensor, "N 4 4"] = poses[:T_in]
        target_Ts: Float[Tensor, "N 4 4"] = poses[T_in : self.num_sample_views]

        input_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[:T_in]
        target_plucker_rays: Float[Tensor, "N 6 H W"] = plucker_rays[T_in : self.num_sample_views]

        input_rays: Float[Tensor, "N 6 H W"] = rays_od[:T_in]
        target_rays: Float[Tensor, "N 6 H W"] = rays_od[T_in : self.num_sample_views]

        # shuffled_indices = torch.randperm(self.num_sample_views)
        shuffled_indices = torch.arange(self.num_sample_views)
        input_indices = shuffled_indices[:T_in]
        target_indices = shuffled_indices[T_in:]

        data = {}
        data["dataset"] = "structured3d"
        data["room_uid"] = room_uid
        data["image_input"] = input_images
        data["image_target"] = target_images
        data["depth_input"] = input_depths
        data["depth_target"] = target_depths
        if self.use_normal:
            data["normal_task_embeddings"] = normal_task_embeddings

        if self.use_semantic:
            data["semantic_task_embeddings"] = semantic_task_embeddings

        if self.use_layout_prior or self.use_layout_prior_from_p3d:
            data["semantic_layout_input"] = render_semantics[:T_in]
            data["semantic_layout_target"] = render_semantics[T_in : self.num_sample_views]
            if not self.use_scene_coord_map:
                normalized_render_depths = normalized_render_depths.repeat(1, 3, 1, 1)
                data["depth_layout_input"] = normalized_render_depths[:T_in]
                data["depth_layout_target"] = normalized_render_depths[T_in : self.num_sample_views]
            else:
                data["depth_layout_input"] = layout_scene_coord_maps[:T_in]
                data["depth_layout_target"] = layout_scene_coord_maps[T_in : self.num_sample_views]
            data["layout_sem_task_embeddings"] = layout_sem_task_embeddings
            data["layout_depth_task_embeddings"] = layout_depth_task_embeddings

        data["pose_out"] = target_Ts
        data["pose_in"] = cond_Ts
        data["plucker_rays_input"] = input_plucker_rays
        data["plucker_rays_target"] = target_plucker_rays
        data["rays_input"] = input_rays
        data["rays_target"] = target_rays
        data["color_task_embeddings"] = color_task_embeddings
        data["depth_task_embeddings"] = depth_task_embeddings
        data["depth_min"] = coord_min
        data["depth_max"] = coord_max
        data["scene_scale"] = curr_scene_scale
        data["input_indices"] = input_indices
        data["output_indices"] = target_indices
        if self.return_metric_data:
            data["pose_metric_input"] = metric_poses[:T_in]
            data["pose_metric_target"] = metric_poses[T_in : self.num_sample_views]
            if self.use_layout_prior or self.use_layout_prior_from_p3d:
                data["layout_depth_metric_input"] = render_depths[:T_in]
                data["layout_depth_metric_target"] = render_depths[T_in : self.num_sample_views]
        data["intrinsic"] = intrinsic_mat
        if self.is_validation and controlnet_rgbs is not None:
            data["controlnet_image_input"] = controlnet_rgbs[:T_in]
            data["controlnet_image_target"] = controlnet_rgbs[T_in : self.num_sample_views]        
        return data
    
    def inner_get_item(self, index: int) -> Dict:
        dataset_name, sample_room_path = list(self.samples[index].keys())[0], list(self.samples[index].values())[0]
        data = {}
        if "spatialgen" == dataset_name:
            # spatialgen dataset
                data = self.get_spatialgen_persp_item(sample_room_path, depth_scale=1000.0)
        elif "hypersim" == dataset_name:
            # hypersim dataset
            data = self.get_hypersim_item(sample_room_path, depth_scale=1.0)
        
        elif "structured3d" == dataset_name:
            # structured3d dataset
            data = self.get_structured3d_item(sample_room_path)
            
        return data

    def __getitem__(self, index):

        dataset_name, sample_room_path = list(self.samples[index].keys())[0], list(self.samples[index].values())[0]
        try:
            return self.inner_get_item(index)
        except Exception as e:
            print(f"[DEBUG-DATASET] Error when loading {sample_room_path}")
            print(f"[DEBUG-DATASET] Error: {e}")
            return self.inner_get_item(index + 1)
            raise e

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.image_height, "width": self.image_width})
        return batch


if __name__=="__main__":
    from tqdm import tqdm
    from icecream import ic
    import open3d as o3d
    import matplotlib.pyplot as plt
    from src.utils.misc import get_device, todevice

    device = torch.device("cpu")
    spatialgen_persp_data_dir = "/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k"
    hypersim_data_dir = None
    st3d_data_dir = None
    spatialgen_split_file = "/data-nas/data/dataset/qunhe/PanoRoom/roomverse_data/processed_data_8k/8k_perspective_trains.txt"
    
    T_in = 1
    T_out = 7
    total_view = T_in + T_out
    image_width = 512
    use_metric_depth = False
    use_scene_coord_map = True
    use_layout_prior = True
    use_layout_prior_from_p3d = True
    train_dataset = MixDataset(
        dataset_names=['spatialgen'],
        spatialgen_data_dir=spatialgen_persp_data_dir,
        hypersim_data_dir=hypersim_data_dir,
        structured3d_data_dir=st3d_data_dir,
        split_filepath=spatialgen_split_file,
        image_height=image_width,
        image_width=image_width,
        T_in=T_in,
        total_view=total_view,
        validation=False,
        sampler_type="spiral",
        use_normal=False,
        use_semantic=False,
        use_metric_depth=use_metric_depth,
        use_scene_coord_map=use_scene_coord_map,
        use_layout_prior=use_layout_prior,
        use_layout_prior_from_p3d=use_layout_prior_from_p3d,
        return_metric_data=True
    )
    validation_dataset = MixDataset(
        dataset_names=['spatialgen'],
        spatialgen_data_dir=spatialgen_persp_data_dir,
        hypersim_data_dir=hypersim_data_dir,
        structured3d_data_dir=st3d_data_dir,
        split_filepath=spatialgen_split_file,
        image_height=image_width,
        image_width=image_width,
        T_in=T_in,
        total_view=total_view,
        validation=True,
        sampler_type="spiral",
        use_normal=True,
        use_semantic=True,
        use_metric_depth=use_metric_depth,
        use_scene_coord_map=use_scene_coord_map,
        use_layout_prior=use_layout_prior,
        use_layout_prior_from_p3d = use_layout_prior_from_p3d,
        return_metric_data=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        collate_fn=validation_dataset.collate,
    )
    val_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
        collate_fn=validation_dataset.collate,
    )

    print(f"MixtureDataset contains {len(val_dataloader)} scenes")

    output_dir = "./debug_mix"
    os.makedirs(output_dir, exist_ok=True)
                
    pbar = tqdm(total=len(val_dataloader))
    for batch in tqdm(val_dataloader):
        pbar.update(1)
        batch = todevice(batch, device)
        
        dataset_type = batch["dataset"][0]

        ic(batch["room_uid"][0])
        
        abs_min_depth = validation_dataset.DEPTH_MIN
        abs_max_depth = validation_dataset.DEPTH_MAX
        min_depth = float(batch["depth_min"][0])
        max_depth = float(batch["depth_max"][0])
        scene_scale = float(batch["scene_scale"][0])
        intrinsic = batch["intrinsic"][0]
        normalized_focal_len = 2.0 * intrinsic[0, 0].cpu().numpy() / image_width
        ic(abs_min_depth, abs_max_depth, min_depth, max_depth, scene_scale, normalized_focal_len)

        input_images = batch["image_input"][0]
        input_scms = batch["depth_input"][0]
        
        if use_layout_prior:
            input_render_layout_scms = batch["depth_layout_input"][0]
            target_render_layout_scms = batch["depth_layout_target"][0]
        
        if dataset_type == "spatialgen":
            input_normals = batch["normal_input"][0]
            input_semantics = batch["semantic_input"][0]
        elif dataset_type == "hypersim":
            input_semantics = batch["semantic_input"][0]
            
        target_images = batch["image_target"][0]
        target_scms = batch["depth_target"][0]
        
        if dataset_type == "spatialgen":
            target_normals = batch["normal_target"][0]
            target_semantics = batch["semantic_target"][0]
        elif dataset_type == "hypersim":
            target_semantics = batch["semantic_target"][0]
            
        if use_layout_prior:
            input_layout_semantics = batch["semantic_layout_input"][0]
            target_layout_semantics = batch["semantic_layout_target"][0]

        pose_in = batch["pose_in"][0]
        poses_out = batch["pose_out"][0]
        ic(pose_in.shape, poses_out.shape)
        metric_pose_in = batch["pose_metric_input"][0]
        metric_pose_out = batch["pose_metric_target"][0]
        metric_poses = torch.cat([metric_pose_in, metric_pose_out], dim=0)
        ic(metric_poses[:,:3,3].min(), metric_poses[:,:3,3].max())
        
        rays_o_in = batch["rays_input"][0][:, 0:3]
        rays_d_in = batch["rays_input"][0][:, 3:6]
        rays_o_out = batch["rays_target"][0][:, 0:3]
        rays_d_out = batch["rays_target"][0][:, 3:6]

        intrinsic_matrix = batch["intrinsic"][0]
        ic(intrinsic_matrix)

        num_input_views = input_images.shape[0]
        num_out_views = target_images.shape[0]

        input_ply = o3d.geometry.PointCloud()
        input_layout_ply = o3d.geometry.PointCloud()

        cam_intrinsic = np.array(
            [
                [image_width / 2.0, 0, image_width / 2.0],
                [0, image_width / 2.0, image_width / 2.0],
                [0, 0, 1],
            ]
        )
        # create a camera
        camera = o3d.camera.PinholeCameraIntrinsic()
        camera.set_intrinsics(
            image_width,
            image_width,
            cam_intrinsic[0, 0],
            cam_intrinsic[1, 1],
            cam_intrinsic[0, 2],
            cam_intrinsic[1, 2],
        )
        
        vis_rgbs = []
        vis_depths = []
        vis_semantics = []
        vis_layout_depths = []
        vis_layout_semantics = []
        vis_pointmaps = []

        def descale_depth(depth, min_depth, max_depth):
            depth = depth * (max_depth - min_depth) + min_depth
            return depth
        
        for i in range(num_input_views):
            input_view_rgb = ((input_images[i, :, :, :] + 1) / 2).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((input_view_rgb * 255).astype(np.uint8)).save(f"{output_dir}/input_cam{i}.png")
            if dataset_type == "spatialgen":
                input_view_normal = input_normals[i].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(((input_view_normal + 1) / 2 * 255.0).astype(np.uint8)).save(f"{output_dir}/input_cam{i}_normal.png")
                input_view_semantic = input_semantics[i].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(((input_view_semantic + 1) / 2 * 255.0).astype(np.uint8)).save(f"{output_dir}/input_cam{i}_semantic.png")
                if use_layout_prior:
                    input_view_layout_semantic = input_layout_semantics[i].permute(1, 2, 0).cpu().numpy()
                    input_view_layout_semantic = ((input_view_layout_semantic + 1) / 2 * 255.0).astype(np.uint8)
                    Image.fromarray(input_view_layout_semantic).save(f"{output_dir}/input_cam{i}_layout_semantic.png")
            elif dataset_type == "hypersim":
                input_view_semantic = input_semantics[i].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(((input_view_semantic + 1) / 2 * 255.0).astype(np.uint8)).save(f"{output_dir}/input_cam{i}_semantic.png")
                input_view_layout_semantic = input_layout_semantics[i].permute(1, 2, 0).cpu().numpy()
                input_view_layout_semantic = ((input_view_layout_semantic + 1) / 2 * 255.0).astype(np.uint8)
                Image.fromarray(input_view_layout_semantic).save(f"{output_dir}/input_cam{i}_layout_semantic.png")
            elif dataset_type == "structured3d":
                input_view_layout_semantic = input_layout_semantics[i].permute(1, 2, 0).cpu().numpy()
                input_view_layout_semantic = ((input_view_layout_semantic + 1) / 2 * 255.0).astype(np.uint8)
                Image.fromarray(input_view_layout_semantic).save(f"{output_dir}/input_cam{i}_layout_semantic.png")
                input_view_semantic = input_layout_semantics[i].permute(1, 2, 0).cpu().numpy()
            # descale scene coordinate map to be metric
            input_view_scm = input_scms[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            input_view_scm = descale_depth(input_view_scm*0.5 + 0.5, min_depth, max_depth) * scene_scale
            if use_layout_prior:
                input_view_layout_scm = input_render_layout_scms[i, :, :, :].permute(1, 2, 0).cpu().numpy()
                input_view_layout_scm = descale_depth(input_view_layout_scm*0.5 + 0.5, min_depth, max_depth) * scene_scale
            # visualize depth
            vis_input_depth = ((input_view_scm * 0.5 + 0.5) * 255).astype(np.uint8)
            Image.fromarray(vis_input_depth).save(f"{output_dir}/input_cam{i}_depth.png")
            if use_layout_prior:
                vis_input_layout_depth = ((input_view_layout_scm * 0.5 + 0.5) * 255).astype(np.uint8)
                Image.fromarray(vis_input_layout_depth).save(f"{output_dir}/input_cam{i}_layout_depth.png")
            
            ply_cam0_points = input_view_scm
            o3d_ply_cam0 = o3d.geometry.PointCloud()
            o3d_ply_cam0.points = o3d.utility.Vector3dVector(ply_cam0_points.reshape(-1, 3))
            o3d_ply_cam0.colors = o3d.utility.Vector3dVector(input_view_rgb.reshape(-1, 3))
            o3d.io.write_point_cloud(f"{output_dir}/input_cam{i}.ply", o3d_ply_cam0)
            input_ply += o3d_ply_cam0
            
            if use_layout_prior:
                ply_cam0_layout_points = input_view_layout_scm
                layout_ply_cam0 = o3d.geometry.PointCloud()
                layout_ply_cam0.points = o3d.utility.Vector3dVector(ply_cam0_layout_points.reshape(-1, 3))
                layout_ply_cam0.colors = o3d.utility.Vector3dVector(input_view_rgb.reshape(-1, 3))
                o3d.io.write_point_cloud(f"{output_dir}/input_layout_cam{i}.ply", layout_ply_cam0)
                input_layout_ply += layout_ply_cam0

            cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(pose_in[i].cpu().numpy()), scale=0.08)
            cam_lines.paint_uniform_color([1, 0, 0])
            o3d.io.write_line_set(os.path.join(output_dir, f"input_cam_{i}.ply"), cam_lines)
            
            # save pointmaps into scene coordinate maps
            ply_cam0_points = ply_cam0_points.reshape(input_view_rgb.shape[0], input_view_rgb.shape[1], 3)
            # convert to [0, 1]
            ply_cam0_points = (ply_cam0_points - np.min(ply_cam0_points)) / (np.max(ply_cam0_points) - np.min(ply_cam0_points))
            
            vis_rgbs.append(input_view_rgb)
            vis_depths.append(vis_input_depth)
            vis_semantics.append(input_view_semantic * 0.5 + 0.5)
            if use_layout_prior:
                vis_layout_depths.append(vis_input_layout_depth)
                vis_layout_semantics.append(input_view_layout_semantic)
            vis_pointmaps.append(ply_cam0_points)

        o3d.io.write_point_cloud(f"{output_dir}/input_pointcloud.ply", input_ply)
        o3d.io.write_point_cloud(f"{output_dir}/input_layout_pointcloud.ply", input_layout_ply)

        target_ply = o3d.geometry.PointCloud()
        target_layout_ply = o3d.geometry.PointCloud()
        
        for i in range(num_out_views):
            sample_view_rgb = ((target_images[i, :, :, :] + 1) / 2).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((sample_view_rgb * 255).astype(np.uint8)).save(f"{output_dir}/target_cam{i}.png")
            if dataset_type == "spatialgen":
                sample_view_normal = target_normals[i].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(((sample_view_normal + 1) / 2 * 255.0).astype(np.uint8)).save(f"{output_dir}/target_cam{i}_normal.png")
                sample_view_semantic = target_semantics[i].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(((sample_view_semantic + 1) / 2 * 255.0).astype(np.uint8)).save(f"{output_dir}/target_cam{i}_semantic.png")
                if use_layout_prior:
                    sample_view_layout_semantics = target_layout_semantics[i].permute(1, 2, 0).cpu().numpy()
                    sample_view_layout_semantics = ((sample_view_layout_semantics + 1) / 2 * 255.0).astype(np.uint8)
                    Image.fromarray(sample_view_layout_semantics).save(f"{output_dir}/target_cam{i}_layout_semantic.png")
            elif dataset_type == "hypersim":
                sample_view_semantic = target_semantics[i].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(((sample_view_semantic + 1) / 2 * 255.0).astype(np.uint8)).save(f"{output_dir}/target_cam{i}_semantic.png")
                sample_view_layout_semantics = target_layout_semantics[i].permute(1, 2, 0).cpu().numpy()
                sample_view_layout_semantics = ((sample_view_layout_semantics + 1) / 2 * 255.0).astype(np.uint8)
                Image.fromarray(sample_view_layout_semantics).save(f"{output_dir}/target_cam{i}_layout_semantic.png")
            elif dataset_type == "structured3d":
                sample_view_layout_semantics = target_layout_semantics[i].permute(1, 2, 0).cpu().numpy()
                sample_view_layout_semantics = ((sample_view_layout_semantics + 1) / 2 * 255.0).astype(np.uint8)
                Image.fromarray(sample_view_layout_semantics).save(f"{output_dir}/target_cam{i}_layout_semantic.png")
                sample_view_semantic = target_layout_semantics[i].permute(1, 2, 0).cpu().numpy()    
            
            sample_view_scm = target_scms[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            sample_view_scm = descale_depth(sample_view_scm*0.5 + 0.5, min_depth, max_depth) * scene_scale
            if use_layout_prior:
                sample_view_layout_scm = target_render_layout_scms[i, :, :, :].permute(1, 2, 0).cpu().numpy()
                sample_view_layout_scm = descale_depth(sample_view_layout_scm*0.5 + 0.5, min_depth, max_depth) * scene_scale
            # visualize depth
            vis_target_depth = ((sample_view_scm * 0.5 + 0.5) * 255).astype(np.uint8)
            Image.fromarray(vis_target_depth).save(f"{output_dir}/target_cam{i}_depth.png")
            if use_layout_prior:
                vis_tar_layout_depth = ((sample_view_layout_scm * 0.5 + 0.5) * 255).astype(np.uint8)
                Image.fromarray(vis_tar_layout_depth).save(f"{output_dir}/target_cam{i}_layout_depth.png")

            ply_cam0_points = sample_view_scm
            o3d_ply_cam0 = o3d.geometry.PointCloud()
            o3d_ply_cam0.points = o3d.utility.Vector3dVector(ply_cam0_points.reshape(-1, 3))
            o3d_ply_cam0.colors = o3d.utility.Vector3dVector(sample_view_rgb.reshape(-1, 3))
            o3d.io.write_point_cloud(f"{output_dir}/target_cam{i}.ply", o3d_ply_cam0)
            
            if use_layout_prior:
                ply_cam0_layout_points = sample_view_layout_scm
                layout_ply_cam0 = o3d.geometry.PointCloud()
                layout_ply_cam0.points = o3d.utility.Vector3dVector(ply_cam0_layout_points.reshape(-1, 3))
                layout_ply_cam0.colors = o3d.utility.Vector3dVector(sample_view_rgb.reshape(-1, 3))
                o3d.io.write_point_cloud(f"{output_dir}/target_layout_cam{i}.ply", layout_ply_cam0)
            
            cam_lines = o3d.geometry.LineSet.create_camera_visualization(intrinsic=camera, extrinsic=np.linalg.inv(poses_out[i].cpu().numpy()), scale=0.08)
            cam_lines.paint_uniform_color([1, 0, 0])
            o3d.io.write_line_set(os.path.join(output_dir, f"target_cam_{i}.ply"), cam_lines)
            
            target_ply += o3d_ply_cam0
            if use_layout_prior:
                target_layout_ply += layout_ply_cam0

            # save pointmaps into scene coordinate maps
            ply_cam0_points = ply_cam0_points.reshape(sample_view_rgb.shape[0], sample_view_rgb.shape[1], 3)
            ply_cam0_points = (ply_cam0_points - np.min(ply_cam0_points)) / (np.max(ply_cam0_points) - np.min(ply_cam0_points))

            vis_rgbs.append(sample_view_rgb)
            vis_depths.append(vis_target_depth)
            vis_semantics.append(sample_view_semantic * 0.5 + 0.5)
            if use_layout_prior:
                vis_layout_depths.append(vis_tar_layout_depth)
                vis_layout_semantics.append(sample_view_layout_semantics)
            vis_pointmaps.append(ply_cam0_points)

        o3d.io.write_point_cloud(f"{output_dir}/target_pointcloud.ply", target_ply)
        o3d.io.write_point_cloud(f"{output_dir}/target_layout_pointcloud.ply", target_layout_ply)

        # draw rgb, depth, pointmap
        fig, axes = plt.subplots(3, num_input_views + num_out_views)
        plt.axis("off")
        for i in range(num_input_views + num_out_views):
            axes[0, i].imshow(vis_rgbs[i])
            axes[1, i].imshow(vis_layout_semantics[i])
            axes[2, i].imshow(vis_layout_depths[i])
        plt.savefig(f"{output_dir}/rgb_depth_pointmap.png", bbox_inches='tight', dpi=1000)

        break
