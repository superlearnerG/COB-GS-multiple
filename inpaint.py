import json
import math
import os
import random
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from random import randint
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from desk_atlas import DeskAtlasState, PlaneDefinition, _deserialize_desk_atlas_state
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import inverse_sigmoid, safe_state
from utils.loss_utils import ssim
from utils.mask_provider import MultiLabelMaskProvider
from utils.projection_utils import camera_intrinsics_tensor, project_xyz_to_plane_uv
from utils.sh_utils import RGB2SH

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


INPAINT_PHASE = "mod_cob_gs_rgb_inpaint"
CHECKPOINT_VERSION = 1
GAUSSIAN_ATTRS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
)


@dataclass
class RGBInpaintState:
    hole_start_idx: int
    hole_end_idx: int
    hole_init_uv: torch.Tensor
    hole_init_stride_px: int
    completed_texture_path: str


@dataclass
class ViewTargetCacheEntry:
    target_rgb_view: torch.Tensor
    merge_mask_view: torch.Tensor
    valid_mask_view: torch.Tensor
    supervision_mask_view: torch.Tensor
    reproj_rgb_view: torch.Tensor
    removal_rgb_view: torch.Tensor
    removal_gt_rgb_view: Optional[torch.Tensor] = None


@dataclass
class RGBTrainingState:
    target_scales_actual: torch.Tensor
    view_target_cache: Dict[int, ViewTargetCacheEntry]


def _cpu_clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_tree(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_tree(val) for val in value)
    return value


def parse_object_id_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        object_ids: List[int] = []
        for item in value:
            object_ids.extend(parse_object_id_list(item))
        return object_ids

    value = str(value).strip()
    if not value:
        return []

    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid object id list: {value}") from exc
        return parse_object_id_list(parsed)

    return [int(item) for item in value.replace(",", " ").split()]


def parse_single_desk_object_id(raw_value) -> int:
    object_ids = parse_object_id_list(raw_value)
    if len(object_ids) == 0:
        raise ValueError("--desk_object_id is required and must contain exactly one positive id.")
    if len(object_ids) > 1:
        raise ValueError(f"--desk_object_id supports exactly one id, got {object_ids}.")
    object_id = int(object_ids[0])
    if object_id <= 0:
        raise ValueError(f"--desk_object_id must be > 0, got {object_id}.")
    return object_id


def _resolve_model_relative_path(path: Optional[str], model_path: str, default_name: str) -> str:
    if path is None or str(path).strip() == "":
        return os.path.join(model_path, default_name)
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(model_path, path)


def _resolve_atlas_dir(model_path: str, desk_atlas_dir: str) -> str:
    return desk_atlas_dir if os.path.isabs(desk_atlas_dir) else os.path.join(model_path, desk_atlas_dir)


def _require_existing_file(path: str, desc: str) -> str:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"{desc} '{path}' not found.")
    return path


def _require_existing_dir(path: str, desc: str) -> str:
    if path is None or not os.path.isdir(path):
        raise FileNotFoundError(f"{desc} directory '{path}' not found.")
    return path


def _load_rgb_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().to(device=device)


def _resolve_view_image_path(root_dir: str, image_name: str) -> str:
    basename = os.path.basename(str(image_name).strip())
    if not basename:
        raise ValueError("Cannot resolve view image path because camera image_name is empty.")

    exact_path = os.path.join(root_dir, basename)
    if os.path.exists(exact_path):
        return exact_path

    stem, ext = os.path.splitext(basename)
    if not ext:
        for suffix in (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"):
            candidate = os.path.join(root_dir, f"{stem}{suffix}")
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(f"Expected removal_GT image for camera '{image_name}' at: {exact_path}")


def _load_removal_gt_for_camera(removal_gt_dir: str, camera, device: torch.device) -> torch.Tensor:
    image_path = _resolve_view_image_path(removal_gt_dir, camera.image_name)
    image = _load_rgb_image_tensor(image_path, device=device)
    expected_hw = (int(camera.image_height), int(camera.image_width))
    if tuple(image.shape[1:]) != expected_hw:
        raise RuntimeError(
            f"removal_GT resolution mismatch for '{camera.image_name}': expected {expected_hw}, "
            f"got {tuple(int(v) for v in image.shape[1:])} at {image_path}."
        )
    return image


def _load_binary_mask_tensor(
    path: str,
    atlas_hw: Tuple[int, int],
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    with Image.open(path) as image:
        mask_np = np.asarray(image.convert("L"), dtype=np.uint8)

    expected_hw = (int(atlas_hw[0]), int(atlas_hw[1]))
    if tuple(mask_np.shape) != expected_hw:
        raise RuntimeError(
            f"{desc} resolution mismatch: expected {expected_hw}, "
            f"got {tuple(int(v) for v in mask_np.shape)}."
        )
    return torch.from_numpy(mask_np > 0).contiguous().to(device=device)


def _load_desk_atlas_state_from_dir(
    model_path: str,
    desk_atlas_dir: str,
    device: torch.device,
) -> DeskAtlasState:
    atlas_dir = _resolve_atlas_dir(model_path, desk_atlas_dir)
    state_path = _require_existing_file(os.path.join(atlas_dir, "desk_atlas_state.pt"), "desk atlas state")
    return _deserialize_desk_atlas_state(torch.load(state_path, map_location="cpu"), device=device)


def _load_rgb_diffusion_pack(
    model_path: str,
    desk_atlas_dir: str,
    desk_atlas_state: DeskAtlasState,
) -> Dict[str, Any]:
    atlas_dir = _resolve_atlas_dir(model_path, desk_atlas_dir)
    candidates = [
        os.path.join(atlas_dir, "desk_diffusion_pack.pt"),
        os.path.join(atlas_dir, "desk_rgb_diffusion_pack.pt"),
    ]
    pack = None
    for path in candidates:
        if os.path.exists(path):
            pack = torch.load(path, map_location="cpu")
            break

    if not isinstance(pack, dict):
        pack = {}

    if "M_support_visible" not in pack:
        pack["M_support_visible"] = desk_atlas_state.support_visible_mask.detach().cpu()
    if "M_support_footprint" not in pack:
        pack["M_support_footprint"] = desk_atlas_state.support_footprint_mask.detach().cpu()
    return pack


def load_rgb_completion_assets(
    model_path: str,
    desk_atlas_dir: str,
    desk_atlas_state: DeskAtlasState,
    completed_texture_path: Optional[str],
    device: torch.device,
) -> Dict[str, Any]:
    pack = _load_rgb_diffusion_pack(model_path, desk_atlas_dir, desk_atlas_state)
    texture_path = _resolve_model_relative_path(completed_texture_path, model_path, "texture_completed.png")
    _require_existing_file(texture_path, "completed texture")
    completed_texture = _load_rgb_image_tensor(texture_path, device=device)

    atlas_h, atlas_w = (int(desk_atlas_state.atlas_hw[0]), int(desk_atlas_state.atlas_hw[1]))
    if tuple(completed_texture.shape[1:]) != (atlas_h, atlas_w):
        raise RuntimeError(
            f"Completed texture resolution mismatch: expected ({atlas_h}, {atlas_w}), "
            f"got {tuple(int(v) for v in completed_texture.shape[1:])}."
        )

    support_visible_mask = pack["M_support_visible"].bool().to(device=device)
    support_footprint_mask = pack["M_support_footprint"].bool().to(device=device)
    atlas_dir = _resolve_atlas_dir(model_path, desk_atlas_dir)
    manmade_mask_path = os.path.join(atlas_dir, "manmade_mask.png")
    support_mask_source = "support_visible_or_footprint"
    support_mask_path = None
    if os.path.exists(manmade_mask_path):
        support_mask_raw = _load_binary_mask_tensor(
            manmade_mask_path,
            atlas_hw=(atlas_h, atlas_w),
            device=device,
            desc="manmade mask",
        )
        support_mask_source = "manmade_mask"
        support_mask_path = os.path.abspath(manmade_mask_path)
    else:
        support_mask_raw = support_visible_mask | support_footprint_mask
    return {
        "completed_texture": completed_texture,
        "completed_texture_path": os.path.abspath(texture_path),
        "support_visible_mask": support_visible_mask,
        "support_footprint_mask": support_footprint_mask,
        "support_mask_raw": support_mask_raw,
        "support_mask_source": support_mask_source,
        "support_mask_path": support_mask_path,
    }


def load_segmentation_checkpoint(gaussians: GaussianModel, checkpoint_path: str) -> Any:
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    if isinstance(checkpoint, (tuple, list)):
        model_state = checkpoint[0]
        marker = checkpoint[1] if len(checkpoint) > 1 else None
    elif isinstance(checkpoint, dict) and "gaussians" in checkpoint:
        model_state = checkpoint["gaussians"]
        marker = checkpoint.get("iteration")
    else:
        model_state = checkpoint
        marker = None

    if not isinstance(model_state, dict):
        raise ValueError(f"Segmentation checkpoint {checkpoint_path} does not contain a Gaussian state dict.")
    if "object_id" not in model_state:
        raise ValueError(f"Segmentation checkpoint {checkpoint_path} does not contain object_id.")

    gaussians._restore_from_state_dict(model_state)
    gaussians._ensure_object_state()
    return marker


def _ensure_exposure_state(gaussians: GaussianModel, train_cameras: Sequence[Any]) -> None:
    image_names = [cam.image_name for cam in train_cameras]
    if not hasattr(gaussians, "exposure_mapping") or not isinstance(getattr(gaussians, "exposure_mapping"), dict):
        gaussians.exposure_mapping = {name: idx for idx, name in enumerate(image_names)}
    if not hasattr(gaussians, "pretrained_exposures"):
        gaussians.pretrained_exposures = None
    if not hasattr(gaussians, "_exposure") or not torch.is_tensor(getattr(gaussians, "_exposure", None)):
        exposure = torch.eye(3, 4, device=gaussians.get_xyz.device)[None].repeat(len(image_names), 1, 1)
        gaussians._exposure = nn.Parameter(exposure.requires_grad_(True))


def _ensure_mask_state(gaussians: GaussianModel) -> None:
    n_points = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    if getattr(gaussians, "_mask", None) is None or gaussians._mask.shape[0] != n_points:
        gaussians._mask = nn.Parameter(torch.zeros((n_points,), dtype=dtype, device=device), requires_grad=False)
    else:
        gaussians._mask = nn.Parameter(gaussians._mask.detach().to(device=device, dtype=dtype), requires_grad=False)
    gaussians._ensure_object_state()


def _uv_to_pixel_coords(
    uv: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u_min, u_max, v_min, v_max = bbox
    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    x = (uv[:, 0] - u_min) / du * max(width - 1, 1)
    y = (uv[:, 1] - v_min) / dv * max(height - 1, 1)
    return x, y


def _pixel_coords_to_uv(
    x: torch.Tensor,
    y: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> torch.Tensor:
    u_min, u_max, v_min, v_max = bbox
    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    u = u_min + x / max(width - 1, 1) * du
    v = v_min + y / max(height - 1, 1) * dv
    return torch.stack([u, v], dim=-1)


def _compute_pixel_world_size(desk_atlas_state: DeskAtlasState) -> Tuple[float, float]:
    h, w = (int(desk_atlas_state.atlas_hw[0]), int(desk_atlas_state.atlas_hw[1]))
    u_min, u_max, v_min, v_max = desk_atlas_state.uv_bbox
    du = max(float(u_max - u_min), 1e-6)
    dv = max(float(v_max - v_min), 1e-6)
    return du / max(w - 1, 1), dv / max(h - 1, 1)


def _sample_mask_grid_coords(mask: torch.Tensor, stride_px: int) -> torch.Tensor:
    h, w = mask.shape
    stride_px = max(int(stride_px), 1)
    start = stride_px // 2
    ys = torch.arange(start, h, stride_px, device=mask.device)
    xs = torch.arange(start, w, stride_px, device=mask.device)
    if ys.numel() == 0:
        ys = torch.arange(0, h, device=mask.device)
    if xs.numel() == 0:
        xs = torch.arange(0, w, device=mask.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
    coords = coords[mask[coords[:, 0].long(), coords[:, 1].long()]]
    if coords.shape[0] > 0:
        return coords
    fallback = mask.nonzero(as_tuple=False)
    if fallback.shape[0] == 0:
        return fallback
    return fallback[::max(stride_px * stride_px, 1)]


def _dilate_binary_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    radius_px = max(int(radius_px), 0)
    if radius_px <= 0:
        return mask.bool()
    x = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = radius_px * 2 + 1
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius_px)
    return y.squeeze(0).squeeze(0) > 0.5


def _erode_binary_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    radius_px = max(int(radius_px), 0)
    if radius_px <= 0:
        return mask.bool()
    x = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = radius_px * 2 + 1
    y = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=radius_px)
    return y.squeeze(0).squeeze(0) > 0.5


def _fill_internal_holes(mask_bool: np.ndarray) -> Tuple[np.ndarray, int, int]:
    mask_bool = np.asarray(mask_bool, dtype=bool)
    h, w = mask_bool.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue = deque()

    def try_push(y: int, x: int) -> None:
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        if mask_bool[y, x] or exterior[y, x]:
            return
        exterior[y, x] = True
        queue.append((y, x))

    for x in range(w):
        try_push(0, x)
        try_push(h - 1, x)
    for y in range(h):
        try_push(y, 0)
        try_push(y, w - 1)

    while queue:
        y, x = queue.popleft()
        try_push(y - 1, x)
        try_push(y + 1, x)
        try_push(y, x - 1)
        try_push(y, x + 1)

    holes = (~mask_bool) & (~exterior)
    filled = mask_bool | holes
    return filled, int(holes.any()), int(holes.sum())


def repair_and_shrink_binary_mask(
    mask: torch.Tensor,
    shrink_px: float,
    close_kernel_size: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = mask.device
    mask = mask.bool()
    original_pixels = int(mask.sum().item())
    if original_pixels <= 0:
        raise RuntimeError("Support mask is empty; cannot build inpaint supervision.")

    close_kernel_size = max(int(close_kernel_size), 0)
    if close_kernel_size > 1:
        radius = close_kernel_size // 2
        closed = _erode_binary_mask(_dilate_binary_mask(mask, radius), radius)
    else:
        closed = mask
    closed_pixels = int(closed.sum().item())

    filled_np, filled_hole_count, filled_hole_area = _fill_internal_holes(closed.detach().cpu().numpy())
    filled = torch.from_numpy(filled_np).to(device=device).bool()
    filled_pixels = int(filled.sum().item())

    shrink_px = float(shrink_px)
    if shrink_px > 0:
        if distance_transform_edt is not None:
            distance_np = distance_transform_edt(filled.detach().cpu().numpy())
            shrunk_np = distance_np > shrink_px
            shrunk = torch.from_numpy(shrunk_np).to(device=device).bool()
        else:
            shrunk = filled
            for _ in range(max(int(math.ceil(shrink_px)), 1)):
                shrunk = _erode_binary_mask(shrunk, 1)
    elif shrink_px < 0:
        expand_px = abs(shrink_px)
        if distance_transform_edt is not None:
            filled_np = filled.detach().cpu().numpy()
            distance_np = distance_transform_edt(~filled_np)
            expanded_np = filled_np | (distance_np <= expand_px)
            shrunk = torch.from_numpy(expanded_np).to(device=device).bool()
        else:
            shrunk = filled
            for _ in range(max(int(math.ceil(expand_px)), 1)):
                shrunk = _dilate_binary_mask(shrunk, 1)
    else:
        shrunk = filled

    if not bool(shrunk.any().item()):
        operation = "Expanded" if shrink_px < 0 else "Shrunk"
        raise RuntimeError(
            f"{operation} support mask is empty. "
            f"Adjust --desk_support_shrink_px (current value: {shrink_px})."
        )

    return shrunk, {
        "original_pixels": float(original_pixels),
        "closed_pixels": float(closed_pixels),
        "filled_pixels": float(filled_pixels),
        "shrunk_pixels": float(int(shrunk.sum().item())),
        "filled_hole_count": float(filled_hole_count),
        "filled_hole_area": float(filled_hole_area),
        "close_kernel_size": float(close_kernel_size),
        "shrink_px": float(shrink_px),
    }


def _build_init_sampling_mask(
    support_footprint_mask: torch.Tensor,
    support_mask_raw: torch.Tensor,
    boundary_px: int,
) -> torch.Tensor:
    init_core_mask = support_footprint_mask.bool()
    boundary_band = _dilate_binary_mask(init_core_mask, boundary_px) & (~init_core_mask) & support_mask_raw.bool()
    return init_core_mask | boundary_band


def rotation_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    r11, r22, r33 = rotation[:, 0, 0], rotation[:, 1, 1], rotation[:, 2, 2]
    qw = torch.sqrt((1.0 + r11 + r22 + r33).clamp_min(1e-7)) * 0.5
    qx = (rotation[:, 2, 1] - rotation[:, 1, 2]) / (4.0 * qw).clamp_min(1e-7)
    qy = (rotation[:, 0, 2] - rotation[:, 2, 0]) / (4.0 * qw).clamp_min(1e-7)
    qz = (rotation[:, 1, 0] - rotation[:, 0, 1]) / (4.0 * qw).clamp_min(1e-7)
    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)
    return F.normalize(quaternion, dim=-1)


def _plane_basis_to_quaternion(
    plane: PlaneDefinition,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    e1 = F.normalize(plane.e1.to(device=device, dtype=torch.float32), dim=0)
    e2 = F.normalize(plane.e2.to(device=device, dtype=torch.float32), dim=0)
    normal = F.normalize(plane.normal.to(device=device, dtype=torch.float32), dim=0)
    rotation = torch.stack([e1, e2, normal], dim=1).unsqueeze(0).expand(int(count), -1, -1)
    return rotation_to_quaternion(rotation).to(device=device, dtype=dtype)


def initialize_hole_gaussians(
    gaussians: GaussianModel,
    desk_atlas_state: DeskAtlasState,
    completed_texture: torch.Tensor,
    hole_init_stride_px: int,
    boundary_px: int,
    support_footprint_mask: torch.Tensor,
    support_mask_raw: torch.Tensor,
    desk_object_id: int,
) -> Tuple[RGBInpaintState, torch.Tensor]:
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    init_sampling_mask = _build_init_sampling_mask(
        support_footprint_mask=support_footprint_mask.to(device=device).bool(),
        support_mask_raw=support_mask_raw.to(device=device).bool(),
        boundary_px=boundary_px,
    )
    coords = _sample_mask_grid_coords(init_sampling_mask, hole_init_stride_px)
    if coords.shape[0] == 0:
        raise RuntimeError("Initialization support footprint mask is empty; cannot initialize hole gaussians.")

    ys = coords[:, 0].float()
    xs = coords[:, 1].float()
    uv = _pixel_coords_to_uv(
        x=xs,
        y=ys,
        bbox=desk_atlas_state.uv_bbox,
        height=int(desk_atlas_state.atlas_hw[0]),
        width=int(desk_atlas_state.atlas_hw[1]),
    )

    plane = desk_atlas_state.plane
    xyz = plane.origin[None] + uv[:, 0:1] * plane.e1[None] + uv[:, 1:2] * plane.e2[None]
    sampled_rgb = completed_texture[:, coords[:, 0].long(), coords[:, 1].long()].permute(1, 0).contiguous()
    n_new = int(sampled_rgb.shape[0])

    pixel_world_x, pixel_world_y = _compute_pixel_world_size(desk_atlas_state)
    scale_x = max(0.5 * float(hole_init_stride_px) * pixel_world_x, 1e-5)
    scale_y = max(0.5 * float(hole_init_stride_px) * pixel_world_y, 1e-5)
    scale_z = max(0.1 * min(scale_x, scale_y), 1e-6)
    target_scales_actual = torch.tensor([scale_x, scale_y, scale_z], dtype=dtype, device=device)

    scaling = torch.log(target_scales_actual[None].expand(n_new, -1))
    rotation = _plane_basis_to_quaternion(plane, n_new, device=device, dtype=dtype)
    opacity = inverse_sigmoid(torch.full((n_new, 1), 0.01, dtype=dtype, device=device))
    features_dc = RGB2SH(sampled_rgb).unsqueeze(1)
    features_rest = torch.zeros(
        (n_new, (gaussians.max_sh_degree + 1) ** 2 - 1, 3),
        dtype=dtype,
        device=device,
    )

    old_count = int(gaussians.get_xyz.shape[0])
    old_tmp_radii = gaussians.tmp_radii
    gaussians.tmp_radii = torch.zeros((old_count,), dtype=dtype, device=device)
    gaussians.densification_postfix(
        xyz.to(device=device, dtype=dtype),
        features_dc.to(device=device, dtype=dtype),
        features_rest,
        opacity,
        scaling,
        rotation,
        torch.zeros((n_new,), dtype=dtype, device=device),
    )
    gaussians.tmp_radii = old_tmp_radii
    new_count = int(gaussians.get_xyz.shape[0])

    with torch.no_grad():
        if getattr(gaussians, "_mask", None) is None or gaussians._mask.shape[0] != old_count:
            old_mask = torch.zeros((old_count,), dtype=dtype, device=device)
        else:
            old_mask = gaussians._mask.detach().to(device=device, dtype=dtype)
        gaussians._mask = nn.Parameter(
            torch.cat([old_mask, torch.zeros((n_new,), dtype=dtype, device=device)], dim=0),
            requires_grad=False,
        )
        gaussians.object_id[old_count:new_count] = int(desk_object_id)
        gaussians.object_score[old_count:new_count] = 1.0

    return (
        RGBInpaintState(
            hole_start_idx=old_count,
            hole_end_idx=new_count,
            hole_init_uv=uv.detach().clone(),
            hole_init_stride_px=int(hole_init_stride_px),
            completed_texture_path="",
        ),
        target_scales_actual,
    )


def _camera_c2w(camera) -> torch.Tensor:
    c2w = getattr(camera, "c2w", None)
    if c2w is not None:
        return c2w
    return torch.inverse(camera.world_view_transform.transpose(0, 1))


def _project_atlas_tensor_to_view(
    camera,
    atlas_tensor: torch.Tensor,
    plane: PlaneDefinition,
    uv_bbox: Tuple[float, float, float, float],
    atlas_hw: Tuple[int, int],
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = camera.camera_center.device
    atlas = atlas_tensor.to(device=device, dtype=torch.float32)
    image_h = int(camera.image_height)
    image_w = int(camera.image_width)
    channels = int(atlas.shape[0])
    view_out = torch.zeros((channels, image_h, image_w), dtype=atlas.dtype, device=device)
    view_valid = torch.zeros((image_h, image_w), dtype=torch.bool, device=device)
    if image_h <= 0 or image_w <= 0 or atlas.numel() == 0:
        return view_out, view_valid

    atlas_h, atlas_w = (int(atlas_hw[0]), int(atlas_hw[1]))
    if int(atlas.shape[1]) != atlas_h or int(atlas.shape[2]) != atlas_w:
        raise RuntimeError(
            "Atlas tensor shape mismatch inside projection: "
            f"expected (*, {atlas_h}, {atlas_w}), got {tuple(int(v) for v in atlas.shape)}."
        )

    plane_normal = plane.normal.to(device=device, dtype=torch.float32)
    plane_d = plane.d.to(device=device, dtype=torch.float32)
    fx, fy, cx, cy = camera_intrinsics_tensor(camera, device=device, dtype=torch.float32)
    c2w_rot = _camera_c2w(camera).to(device=device, dtype=torch.float32)[:3, :3]
    camera_origin = camera.camera_center.to(device=device, dtype=torch.float32)
    origin_term = torch.dot(camera_origin, plane_normal) + plane_d
    xs_full = torch.arange(image_w, device=device, dtype=torch.float32)

    for y_start in range(0, image_h, 128):
        y_end = min(y_start + 128, image_h)
        chunk_h = y_end - y_start
        ys = torch.arange(y_start, y_end, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs_full, indexing="ij")
        dirs_cam = torch.stack(
            [
                (grid_x.reshape(-1) - cx) / fx,
                (grid_y.reshape(-1) - cy) / fy,
                torch.ones((chunk_h * image_w,), device=device, dtype=torch.float32),
            ],
            dim=-1,
        )
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        dirs_world = (c2w_rot @ dirs_cam.t()).t()
        denom = dirs_world @ plane_normal
        valid = torch.abs(denom) > 1e-7
        chunk_valid = torch.zeros((chunk_h * image_w,), dtype=torch.bool, device=device)
        chunk_out = torch.zeros((channels, chunk_h * image_w), dtype=atlas.dtype, device=device)

        if valid.any():
            dirs_world_valid = dirs_world[valid]
            denom_valid = denom[valid]
            t = -origin_term / denom_valid
            hit = t > 0
            if hit.any():
                points = camera_origin[None] + dirs_world_valid[hit] * t[hit][:, None]
                uv = project_xyz_to_plane_uv(points, plane)
                pix_x, pix_y = _uv_to_pixel_coords(uv, uv_bbox, atlas_h, atlas_w)
                in_atlas = (
                    (pix_x >= 0.0)
                    & (pix_x <= max(atlas_w - 1, 0))
                    & (pix_y >= 0.0)
                    & (pix_y <= max(atlas_h - 1, 0))
                )
                valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
                hit_idx = valid_idx[hit]
                chunk_valid[hit_idx] = in_atlas

                if in_atlas.any():
                    grid_x_norm = pix_x[in_atlas] / max(atlas_w - 1, 1) * 2.0 - 1.0
                    grid_y_norm = pix_y[in_atlas] / max(atlas_h - 1, 1) * 2.0 - 1.0
                    sample_grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1).view(1, -1, 1, 2)
                    sampled = F.grid_sample(
                        atlas.unsqueeze(0),
                        sample_grid,
                        mode=mode,
                        padding_mode="zeros",
                        align_corners=True,
                    ).squeeze(0).squeeze(-1)
                    chunk_out[:, hit_idx[in_atlas]] = sampled

        view_out[:, y_start:y_end, :] = chunk_out.view(channels, chunk_h, image_w)
        view_valid[y_start:y_end, :] = chunk_valid.view(chunk_h, image_w)

    return view_out, view_valid


def project_rgb_atlas_to_view(
    camera,
    atlas_tensor: torch.Tensor,
    plane: PlaneDefinition,
    uv_bbox: Tuple[float, float, float, float],
    atlas_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    view_value, view_valid = _project_atlas_tensor_to_view(
        camera=camera,
        atlas_tensor=atlas_tensor,
        plane=plane,
        uv_bbox=uv_bbox,
        atlas_hw=atlas_hw,
        mode="bilinear",
    )
    return {"value": view_value, "valid_mask": view_valid}


def project_binary_atlas_mask_to_view(
    camera,
    atlas_mask: torch.Tensor,
    plane: PlaneDefinition,
    uv_bbox: Tuple[float, float, float, float],
    atlas_hw: Tuple[int, int],
) -> torch.Tensor:
    view_mask, view_valid = _project_atlas_tensor_to_view(
        camera=camera,
        atlas_tensor=atlas_mask.float().unsqueeze(0),
        plane=plane,
        uv_bbox=uv_bbox,
        atlas_hw=atlas_hw,
        mode="nearest",
    )
    return view_valid & (view_mask.squeeze(0) > 0.5)


def _view_target_entry_to_device(entry: ViewTargetCacheEntry, device: torch.device) -> ViewTargetCacheEntry:
    return ViewTargetCacheEntry(
        target_rgb_view=entry.target_rgb_view.to(device=device),
        merge_mask_view=entry.merge_mask_view.to(device=device),
        valid_mask_view=entry.valid_mask_view.to(device=device),
        supervision_mask_view=entry.supervision_mask_view.to(device=device),
        reproj_rgb_view=entry.reproj_rgb_view.to(device=device),
        removal_rgb_view=entry.removal_rgb_view.to(device=device),
        removal_gt_rgb_view=entry.removal_gt_rgb_view.to(device=device) if entry.removal_gt_rgb_view is not None else None,
    )


def build_view_target_cache(
    scene: Scene,
    gaussians: GaussianModel,
    desk_atlas_state: DeskAtlasState,
    completed_texture: torch.Tensor,
    merge_atlas_mask: torch.Tensor,
    pipe: PipelineParams,
    opt: OptimizationParams,
    background: torch.Tensor,
    reference_filter: torch.Tensor,
    removal_gt_dir: Optional[str] = None,
) -> Dict[int, ViewTargetCacheEntry]:
    cache: Dict[int, ViewTargetCacheEntry] = {}
    if removal_gt_dir is not None:
        _require_existing_dir(removal_gt_dir, "removal_GT")
    with torch.no_grad():
        for cam in tqdm(scene.getTrainCameras(), desc="BuildRGBInpaintTargets"):
            removal_pkg = render(cam, gaussians, pipe, background, opt, gaussian_filter=reference_filter)
            removal_rgb = removal_pkg["render"].detach().clamp(0.0, 1.0)
            removal_gt_rgb = None
            if removal_gt_dir is not None:
                removal_gt_rgb = _load_removal_gt_for_camera(removal_gt_dir, cam, device=removal_rgb.device)
            reproj = project_rgb_atlas_to_view(
                camera=cam,
                atlas_tensor=completed_texture,
                plane=desk_atlas_state.plane,
                uv_bbox=desk_atlas_state.uv_bbox,
                atlas_hw=desk_atlas_state.atlas_hw,
            )
            merge_mask_view = project_binary_atlas_mask_to_view(
                camera=cam,
                atlas_mask=merge_atlas_mask,
                plane=desk_atlas_state.plane,
                uv_bbox=desk_atlas_state.uv_bbox,
                atlas_hw=desk_atlas_state.atlas_hw,
            )
            valid_mask = merge_mask_view & reproj["valid_mask"]
            supervision_mask = (~merge_mask_view) | valid_mask
            if getattr(cam, "alpha_mask", None) is not None:
                supervision_mask = supervision_mask & (cam.alpha_mask.to(device=valid_mask.device).squeeze(0) > 0.0)
            target_rgb = torch.where(valid_mask.unsqueeze(0), reproj["value"], removal_rgb)
            cache[int(cam.uid)] = ViewTargetCacheEntry(
                target_rgb_view=target_rgb.detach().cpu(),
                merge_mask_view=merge_mask_view.detach().cpu(),
                valid_mask_view=valid_mask.detach().cpu(),
                supervision_mask_view=supervision_mask.detach().cpu(),
                reproj_rgb_view=reproj["value"].detach().cpu(),
                removal_rgb_view=removal_rgb.detach().cpu(),
                removal_gt_rgb_view=removal_gt_rgb.detach().cpu() if removal_gt_rgb is not None else None,
            )
    return cache


def _masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"RGB loss shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    weight = mask.float()
    denom = weight.sum() * pred.shape[0] + 1e-8
    return (torch.abs(pred - target) * weight.unsqueeze(0)).sum() / denom


def compute_rgb_inpaint_loss(
    image: torch.Tensor,
    target_rgb: torch.Tensor,
    supervision_mask: torch.Tensor,
    opt: OptimizationParams,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ll1 = _masked_l1_loss(image, target_rgb, supervision_mask)
    mask3 = supervision_mask.unsqueeze(0).expand_as(image)
    image_for_ssim = torch.where(mask3, image, target_rgb.detach())
    target_for_ssim = target_rgb
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image_for_ssim.unsqueeze(0), target_for_ssim.unsqueeze(0))
    else:
        ssim_value = ssim(image_for_ssim, target_for_ssim)
    loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    return loss, {
        "loss_l1": float(ll1.item()),
        "loss_total": float(loss.item()),
        "ssim": float(ssim_value.item()),
        "supervision_mean": float(supervision_mask.float().mean().item()),
    }


def _normalize_removal_gt_ratio(ratio: float) -> float:
    ratio = float(ratio)
    if not math.isfinite(ratio):
        raise ValueError(f"--removal_GT_ratio must be finite, got {ratio}.")
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"--removal_GT_ratio must be in [0, 1], got {ratio}.")
    return ratio


def _removal_gt_schedule_ratio(ratio: float) -> Fraction:
    ratio = _normalize_removal_gt_ratio(ratio)
    return Fraction(ratio).limit_denominator(100)


def _use_removal_gt_for_iteration(iteration: int, source_iteration: int, schedule: Fraction) -> bool:
    if schedule.numerator <= 0:
        return False
    if schedule.numerator >= schedule.denominator:
        return True
    local_step = max(int(iteration) - int(source_iteration), 1)
    return ((local_step - 1) % int(schedule.denominator)) < int(schedule.numerator)


def _dilate_binary_view_mask(mask: torch.Tensor, dilation_px: int) -> torch.Tensor:
    mask = mask.bool()
    if int(dilation_px) <= 0:
        return mask
    radius = int(dilation_px)
    return (
        F.max_pool2d(
            mask.float()[None, None],
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
        .squeeze(0)
        .squeeze(0)
        > 0.5
    )


def _union_view_object_mask(
    mask_provider: MultiLabelMaskProvider,
    viewpoint_cam,
    object_ids: Sequence[int],
    dilation_px: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        (int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)),
        dtype=torch.bool,
        device=device,
    )
    for object_id in object_ids:
        mask = mask | mask_provider.get_mask(viewpoint_cam, int(object_id)).to(device=device).squeeze(0).bool()
    mask = _dilate_binary_view_mask(mask, int(dilation_px))
    if getattr(viewpoint_cam, "alpha_mask", None) is not None:
        mask = mask & (viewpoint_cam.alpha_mask.to(device=device).squeeze(0) > 0.0)
    return mask


def _resolve_opacity_recalibration_object_ids(
    gaussians: GaussianModel,
    desk_object_id: int,
    decouple_object_ids: Sequence[int],
    requested_object_ids: Sequence[int],
) -> List[int]:
    assigned_labels = set(_assigned_positive_labels(gaussians))
    if requested_object_ids:
        object_ids = [int(v) for v in requested_object_ids]
    elif decouple_object_ids:
        object_ids = [int(v) for v in decouple_object_ids]
    else:
        object_ids = [label for label in sorted(assigned_labels) if label != int(desk_object_id)]
    object_ids = sorted({int(v) for v in object_ids if int(v) > 0 and int(v) != int(desk_object_id)})
    missing = [object_id for object_id in object_ids if object_id not in assigned_labels]
    if missing:
        raise RuntimeError(
            f"Opacity recalibration requested object ids {missing}, but assigned labels are "
            f"{sorted(assigned_labels)}."
        )
    if not object_ids:
        raise RuntimeError(
            "Opacity recalibration needs at least one non-desk object id. "
            "Pass --opacity_recalibration_object_id or --decouple_object_id."
        )
    return object_ids


def _opacity_logit_bounds(
    min_opacity: float = 1e-4,
    max_opacity: float = 0.995,
) -> Tuple[float, float]:
    min_opacity = min(max(float(min_opacity), 1e-6), 1.0 - 1e-6)
    max_opacity = min(max(float(max_opacity), min_opacity + 1e-6), 1.0 - 1e-6)
    return (
        math.log(min_opacity / (1.0 - min_opacity)),
        math.log(max_opacity / (1.0 - max_opacity)),
    )


def run_opacity_recalibration(
    scene: Scene,
    gaussians: GaussianModel,
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    args,
    desk_object_id: int,
    decouple_object_ids: Sequence[int],
    background: torch.Tensor,
) -> Dict[str, Any]:
    iterations = int(getattr(args, "opacity_recalibration_iterations", 500))
    if iterations <= 0:
        return {"enabled": True, "skipped": True, "reason": "non_positive_iterations"}

    device = gaussians.get_xyz.device
    requested_object_ids = parse_object_id_list(getattr(args, "opacity_recalibration_object_id", None))
    object_ids = _resolve_opacity_recalibration_object_ids(
        gaussians=gaussians,
        desk_object_id=desk_object_id,
        decouple_object_ids=decouple_object_ids,
        requested_object_ids=requested_object_ids,
    )
    object_filter = torch.zeros_like(gaussians.get_object_id, dtype=torch.bool, device=device)
    for object_id in object_ids:
        object_filter = object_filter | (gaussians.get_object_id == int(object_id))
    if int(object_filter.sum().item()) == 0:
        raise RuntimeError(f"Opacity recalibration object ids {object_ids} select no Gaussians.")

    train_cameras = scene.getTrainCameras()
    mask_provider = MultiLabelMaskProvider(
        dataset.mask_root,
        train_cameras,
        target_labels=",".join(str(object_id) for object_id in object_ids),
        object_order=dataset.object_order,
    )

    lr = float(getattr(args, "opacity_recalibration_lr", 0.005))
    dilation_px = int(getattr(args, "opacity_recalibration_dilation_px", 3))
    reg_weight = float(getattr(args, "opacity_recalibration_reg_weight", 0.01))
    initial_opacity = gaussians.get_opacity.detach().clone()
    opacity_optimizer = torch.optim.Adam([gaussians._opacity], lr=lr, eps=1e-15)

    requires_grad_state = {}
    for attr_name in GAUSSIAN_ATTRS:
        param = getattr(gaussians, attr_name, None)
        if param is None or not hasattr(param, "requires_grad"):
            continue
        requires_grad_state[attr_name] = bool(param.requires_grad)
        param.requires_grad_(attr_name == "_opacity")

    last_metrics: Dict[str, float] = {}
    skipped_empty_masks = 0
    logit_min, logit_max = _opacity_logit_bounds()
    viewpoint_stack = None
    try:
        with torch.enable_grad():
            progress_bar = tqdm(range(1, iterations + 1), desc="OpacityRecalibration", miniters=10)
            for step in progress_bar:
                if not viewpoint_stack:
                    viewpoint_stack = train_cameras.copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                opacity_optimizer.zero_grad(set_to_none=True)
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
                image = render_pkg["render"]
                if getattr(viewpoint_cam, "alpha_mask", None) is not None:
                    image = image * viewpoint_cam.alpha_mask.to(device=image.device)

                supervision_mask = _union_view_object_mask(
                    mask_provider=mask_provider,
                    viewpoint_cam=viewpoint_cam,
                    object_ids=object_ids,
                    dilation_px=dilation_px,
                    device=device,
                )
                if int(supervision_mask.sum().item()) == 0:
                    skipped_empty_masks += 1
                    continue

                target_rgb = viewpoint_cam.original_image.to(device=device).clamp(0.0, 1.0)
                loss, metrics = compute_rgb_inpaint_loss(
                    image=image,
                    target_rgb=target_rgb,
                    supervision_mask=supervision_mask,
                    opt=opt,
                )
                opacity_reg = torch.zeros((), dtype=loss.dtype, device=device)
                if reg_weight > 0:
                    opacity_reg = F.l1_loss(gaussians.get_opacity[object_filter], initial_opacity[object_filter])
                    loss = loss + reg_weight * opacity_reg

                loss.backward()
                if gaussians._opacity.grad is not None:
                    gaussians._opacity.grad[~object_filter] = 0
                opacity_optimizer.step()
                with torch.no_grad():
                    gaussians._opacity.data[object_filter] = gaussians._opacity.data[object_filter].clamp(
                        min=logit_min,
                        max=logit_max,
                    )

                last_metrics = {
                    **metrics,
                    "loss_total_with_reg": float(loss.item()),
                    "opacity_reg": float(opacity_reg.item()),
                }
                progress_bar.set_postfix(
                    {
                        "loss": f"{last_metrics['loss_total_with_reg']:.5f}",
                        "l1": f"{last_metrics['loss_l1']:.5f}",
                        "reg": f"{last_metrics['opacity_reg']:.5f}",
                    }
                )
    finally:
        opacity_optimizer.zero_grad(set_to_none=True)
        for attr_name, requires_grad in requires_grad_state.items():
            getattr(gaussians, attr_name).requires_grad_(requires_grad)

    final_opacity = gaussians.get_opacity.detach()
    metadata = {
        "enabled": True,
        "iterations": int(iterations),
        "lr": float(lr),
        "dilation_px": int(dilation_px),
        "reg_weight": float(reg_weight),
        "object_id": [int(object_id) for object_id in object_ids],
        "object_gaussian_count": int(object_filter.sum().item()),
        "skipped_empty_masks": int(skipped_empty_masks),
        "mean_opacity_before": float(initial_opacity[object_filter].mean().item()),
        "mean_opacity_after": float(final_opacity[object_filter].mean().item()),
    }
    metadata.update(last_metrics)
    return metadata


def _mask_parameter_grads(gaussians: GaussianModel, train_filter: torch.Tensor) -> None:
    freeze_filter = ~train_filter.bool()
    for attr_name in GAUSSIAN_ATTRS:
        param = getattr(gaussians, attr_name, None)
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if grad.shape[0] != freeze_filter.shape[0]:
            raise RuntimeError(
                f"Gradient shape mismatch for {attr_name}: grad={tuple(grad.shape)} "
                f"filter={tuple(freeze_filter.shape)}"
            )
        grad[freeze_filter] = 0


def _project_hole_gaussians_back_to_plane(
    gaussians: GaussianModel,
    inpaint_state: RGBInpaintState,
    desk_atlas_state: DeskAtlasState,
    target_scales_actual: torch.Tensor,
) -> None:
    start_idx = int(inpaint_state.hole_start_idx)
    end_idx = int(inpaint_state.hole_end_idx)
    if end_idx <= start_idx:
        return
    with torch.no_grad():
        plane = desk_atlas_state.plane
        hole_xyz = gaussians._xyz.data[start_idx:end_idx]
        hole_uv = project_xyz_to_plane_uv(hole_xyz, plane)
        gaussians._xyz.data[start_idx:end_idx] = (
            plane.origin[None] + hole_uv[:, 0:1] * plane.e1[None] + hole_uv[:, 1:2] * plane.e2[None]
        )
        hole_scales = gaussians.get_scaling[start_idx:end_idx]
        min_scale = 0.5 * target_scales_actual
        max_scale = 2.0 * target_scales_actual
        clamped_scale = torch.maximum(torch.minimum(hole_scales, max_scale.unsqueeze(0)), min_scale.unsqueeze(0))
        gaussians._scaling.data[start_idx:end_idx] = torch.log(clamped_scale.clamp_min(1e-6))


def _save_training_vis(
    model_path: str,
    iteration: int,
    viewpoint_cam,
    render_image: torch.Tensor,
    view_target_entry: ViewTargetCacheEntry,
) -> None:
    vis_dir = os.path.join(model_path, "visualize_inpaint")
    os.makedirs(vis_dir, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(viewpoint_cam.image_name))
    removal_gt_vis = (
        view_target_entry.removal_gt_rgb_view.detach().clamp(0.0, 1.0)
        if view_target_entry.removal_gt_rgb_view is not None
        else torch.zeros_like(render_image.detach())
    )
    grid = make_grid(
        torch.stack(
            [
                render_image.detach().clamp(0.0, 1.0),
                view_target_entry.target_rgb_view.detach().clamp(0.0, 1.0),
                view_target_entry.reproj_rgb_view.detach().clamp(0.0, 1.0),
                view_target_entry.removal_rgb_view.detach().clamp(0.0, 1.0),
                view_target_entry.merge_mask_view.detach().unsqueeze(0).repeat(3, 1, 1).float(),
                view_target_entry.valid_mask_view.detach().unsqueeze(0).repeat(3, 1, 1).float(),
                view_target_entry.supervision_mask_view.detach().unsqueeze(0).repeat(3, 1, 1).float(),
                removal_gt_vis,
            ],
            dim=0,
        ),
        nrow=4,
    )
    save_image(grid, os.path.join(vis_dir, f"{iteration:06d}_{safe_name}.png"))


def _metadata_payload(
    iteration: int,
    source_iteration: int,
    desk_object_id: int,
    decouple_object_ids: Sequence[int],
    excluded_object_ids: Sequence[int],
    inpaint_state: RGBInpaintState,
    desk_atlas_state: DeskAtlasState,
    support_stats: Dict[str, float],
    init_footprint_stats: Dict[str, float],
    train_count: int,
    frozen_count: int,
    opacity_recalibration: Optional[Dict[str, Any]] = None,
    removal_gt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "phase": INPAINT_PHASE,
        "version": CHECKPOINT_VERSION,
        "iteration": int(iteration),
        "source_iteration": int(source_iteration),
        "desk_object_id": int(desk_object_id),
        "decouple_object_id": [int(v) for v in decouple_object_ids],
        "excluded_object_id": [int(v) for v in excluded_object_ids],
        "hole_start_idx": int(inpaint_state.hole_start_idx),
        "hole_end_idx": int(inpaint_state.hole_end_idx),
        "hole_init_stride_px": int(inpaint_state.hole_init_stride_px),
        "completed_texture_path": str(inpaint_state.completed_texture_path),
        "desk_atlas_hw": [int(desk_atlas_state.atlas_hw[0]), int(desk_atlas_state.atlas_hw[1])],
        "desk_atlas_uv_bbox": [float(v) for v in desk_atlas_state.uv_bbox],
        "support_stats": {key: float(value) for key, value in support_stats.items()},
        "init_footprint_stats": {key: float(value) for key, value in init_footprint_stats.items()},
        "train_gaussian_count": int(train_count),
        "frozen_gaussian_count": int(frozen_count),
        "opacity_recalibration": opacity_recalibration or {"enabled": False},
        "removal_GT": removal_gt or {"enabled": False},
    }


def _save_inpaint_outputs(
    model_path: str,
    iteration: int,
    gaussians: GaussianModel,
    metadata: Dict[str, Any],
    save_point_cloud: bool,
    save_checkpoint: bool,
) -> None:
    if save_point_cloud:
        print(f"\n[ITER {iteration}] Saving RGB inpaint point cloud")
        point_cloud_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
        os.makedirs(point_cloud_path, exist_ok=True)
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        torch.save(
            {
                "gaussians": _cpu_clone_tree(gaussians.capture(include_mask=True)),
                "inpaint": metadata,
            },
            os.path.join(point_cloud_path, "point_cloud.pth"),
        )
    if save_checkpoint:
        print(f"\n[ITER {iteration}] Saving RGB inpaint checkpoint")
        torch.save(
            (gaussians.capture(include_mask=True), int(iteration)),
            os.path.join(model_path, f"chkpnt{iteration}.pth"),
        )
    with open(os.path.join(model_path, "inpaint_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)


def _assigned_positive_labels(gaussians: GaussianModel) -> List[int]:
    labels = torch.unique(gaussians.get_object_id.detach())
    return sorted(int(label.item()) for label in labels if int(label.item()) > 0)


def _make_train_filter(gaussians: GaussianModel, desk_object_id: int) -> torch.Tensor:
    object_id = gaussians.get_object_id
    return torch.logical_or(object_id == 0, object_id == int(desk_object_id))


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, args) -> Dict[str, Any]:
    source_iteration = int(args.source_iteration)
    if source_iteration <= 0:
        raise ValueError(f"--source_iteration must be positive, got {source_iteration}.")
    if int(opt.iterations) <= source_iteration:
        raise ValueError(f"--iterations ({opt.iterations}) must be > source_iteration ({source_iteration}).")

    desk_object_id = parse_single_desk_object_id(getattr(args, "desk_object_id", None))
    decouple_object_ids = sorted({int(v) for v in parse_object_id_list(getattr(args, "decouple_object_id", None)) if int(v) > 0})

    opt.include_mask = False
    opt.inpainting = True
    opt.optimizer_type = "default"
    opt.random_background = False

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=source_iteration, shuffle=False)
    device = scene.gaussians.get_xyz.device

    checkpoint_path = getattr(args, "segmentation_checkpoint", None)
    if not checkpoint_path:
        checkpoint_path = os.path.join(dataset.model_path, "multi_object", "final_multi_object.pth")
    checkpoint_path = os.path.abspath(checkpoint_path)
    _require_existing_file(checkpoint_path, "segmentation checkpoint")
    load_segmentation_checkpoint(scene.gaussians, checkpoint_path)
    _ensure_exposure_state(scene.gaussians, scene.getTrainCameras())
    _ensure_mask_state(scene.gaussians)

    if desk_object_id not in _assigned_positive_labels(scene.gaussians):
        raise RuntimeError(
            f"desk_object_id={desk_object_id} is not present in checkpoint labels "
            f"{_assigned_positive_labels(scene.gaussians)}."
        )
    if decouple_object_ids and desk_object_id in decouple_object_ids:
        raise ValueError("--decouple_object_id must not contain --desk_object_id.")

    scene.gaussians.training_setup(opt)
    _ensure_mask_state(scene.gaussians)

    desk_atlas_state = _load_desk_atlas_state_from_dir(dataset.model_path, args.desk_atlas_dir, device=device)
    completed_assets = load_rgb_completion_assets(
        model_path=dataset.model_path,
        desk_atlas_dir=args.desk_atlas_dir,
        desk_atlas_state=desk_atlas_state,
        completed_texture_path=getattr(args, "completed_texture_path", None),
        device=device,
    )
    completed_texture = completed_assets["completed_texture"]
    merged_support_mask, merged_support_stats = repair_and_shrink_binary_mask(
        completed_assets["support_mask_raw"],
        float(args.desk_support_shrink_px),
        close_kernel_size=int(args.desk_support_close_kernel_size),
    )
    init_footprint_mask, init_footprint_stats = repair_and_shrink_binary_mask(
        completed_assets["support_footprint_mask"],
        float(args.desk_support_shrink_px),
        close_kernel_size=int(args.desk_support_close_kernel_size),
    )

    inpaint_state, target_scales_actual = initialize_hole_gaussians(
        gaussians=scene.gaussians,
        desk_atlas_state=desk_atlas_state,
        completed_texture=completed_texture,
        hole_init_stride_px=int(args.hole_init_stride_px),
        boundary_px=int(args.view_supervision_boundary_px),
        support_footprint_mask=init_footprint_mask,
        support_mask_raw=completed_assets["support_mask_raw"],
        desk_object_id=desk_object_id,
    )
    inpaint_state.completed_texture_path = str(completed_assets["completed_texture_path"])

    train_filter = _make_train_filter(scene.gaussians, desk_object_id)
    reference_filter = train_filter.clone()
    reference_filter[int(inpaint_state.hole_start_idx):int(inpaint_state.hole_end_idx)] = False
    excluded_object_ids = [label for label in _assigned_positive_labels(scene.gaussians) if label != int(desk_object_id)]
    removal_gt_enabled = bool(getattr(args, "removal_GT", False))
    removal_gt_ratio = float(getattr(args, "removal_GT_ratio", 0.75))
    removal_gt_schedule = Fraction(0, 1)
    if removal_gt_enabled:
        removal_gt_ratio = _normalize_removal_gt_ratio(removal_gt_ratio)
        removal_gt_schedule = _removal_gt_schedule_ratio(removal_gt_ratio)
    removal_gt_dir = os.path.join(dataset.source_path, "removal_GT") if removal_gt_enabled else None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    training_state = RGBTrainingState(
        target_scales_actual=target_scales_actual,
        view_target_cache=build_view_target_cache(
            scene=scene,
            gaussians=scene.gaussians,
            desk_atlas_state=desk_atlas_state,
            completed_texture=completed_texture,
            merge_atlas_mask=merged_support_mask,
            pipe=pipe,
            opt=opt,
            background=background,
            reference_filter=reference_filter,
            removal_gt_dir=removal_gt_dir,
        ),
    )

    train_count = int(train_filter.sum().item())
    frozen_count = int((~train_filter).sum().item())
    print("[MOD_COB_INPAINT][SETUP] Restoration complete.")
    print(f"[MOD_COB_INPAINT][SETUP] model_path={dataset.model_path}")
    print(f"[MOD_COB_INPAINT][SETUP] source_iteration={source_iteration} target_iteration={opt.iterations}")
    print(f"[MOD_COB_INPAINT][SETUP] segmentation_checkpoint={checkpoint_path}")
    print(f"[MOD_COB_INPAINT][SETUP] completed_texture={inpaint_state.completed_texture_path}")
    if completed_assets["support_mask_path"] is not None:
        print(
            "[MOD_COB_INPAINT][SETUP] "
            f"support_mask_source={completed_assets['support_mask_source']} "
            f"path={completed_assets['support_mask_path']}"
        )
    else:
        print(f"[MOD_COB_INPAINT][SETUP] support_mask_source={completed_assets['support_mask_source']}")
    print(
        "[MOD_COB_INPAINT][SETUP] "
        f"desk_object_id={desk_object_id} train_gaussians={train_count} frozen_gaussians={frozen_count} "
        f"excluded_object_ids={excluded_object_ids}"
    )
    print(
        "[MOD_COB_INPAINT][SETUP] "
        f"hole_index_range=[{inpaint_state.hole_start_idx}, {inpaint_state.hole_end_idx})"
    )
    print(
        "[MOD_COB_INPAINT][SETUP] "
        f"removal_GT_enabled={removal_gt_enabled} "
        f"ratio={removal_gt_ratio:.4f} "
        f"schedule={removal_gt_schedule.numerator}/{removal_gt_schedule.denominator} "
        f"dir={removal_gt_dir or '<disabled>'}"
    )

    viewpoint_stack = None
    progress_bar = tqdm(
        range(source_iteration + 1, int(opt.iterations) + 1),
        desc="ModCOBGSInpaint",
        initial=source_iteration,
        total=int(opt.iterations),
        miniters=10,
    )

    last_metrics: Dict[str, float] = {}
    opacity_recalibration_metadata: Dict[str, Any] = {"enabled": False}
    removal_gt_steps = 0
    constructed_gt_steps = 0
    for iteration in progress_bar:
        scene.gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            scene.gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if (iteration - 1) == int(args.debug_from):
            pipe.debug = True

        scene.gaussians.optimizer.zero_grad(set_to_none=True)
        bg = torch.rand((3), device=device) if opt.random_background else background
        render_pkg = render(viewpoint_cam, scene.gaussians, pipe, bg, opt, gaussian_filter=train_filter)
        image = render_pkg["render"]
        if getattr(viewpoint_cam, "alpha_mask", None) is not None:
            image = image * viewpoint_cam.alpha_mask.to(device=image.device)

        view_target_entry = _view_target_entry_to_device(
            training_state.view_target_cache[int(viewpoint_cam.uid)],
            device,
        )
        use_removal_gt = removal_gt_enabled and _use_removal_gt_for_iteration(
            iteration,
            source_iteration,
            removal_gt_schedule,
        )
        if use_removal_gt:
            if view_target_entry.removal_gt_rgb_view is None:
                raise RuntimeError(f"removal_GT target was not cached for camera '{viewpoint_cam.image_name}'.")
            target_rgb = view_target_entry.removal_gt_rgb_view
            supervision_mask = torch.ones(
                (int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)),
                dtype=torch.bool,
                device=device,
            )
            target_mode = "removal_GT"
            removal_gt_steps += 1
        else:
            target_rgb = view_target_entry.target_rgb_view
            supervision_mask = view_target_entry.supervision_mask_view
            target_mode = "constructed_GT"
            constructed_gt_steps += 1
        loss, metrics = compute_rgb_inpaint_loss(
            image=image,
            target_rgb=target_rgb,
            supervision_mask=supervision_mask,
            opt=opt,
        )
        loss.backward()
        _mask_parameter_grads(scene.gaussians, train_filter)

        with torch.no_grad():
            scene.gaussians.optimizer.step()
            scene.gaussians.optimizer.zero_grad(set_to_none=True)
            _project_hole_gaussians_back_to_plane(
                scene.gaussians,
                inpaint_state,
                desk_atlas_state,
                training_state.target_scales_actual,
            )

            if bool(getattr(args, "save_training_vis", False)) and (
                iteration % int(getattr(args, "save_training_vis_iteration", 1000)) == 0
                or iteration == source_iteration + 1
            ):
                _save_training_vis(
                    dataset.model_path,
                    iteration,
                    viewpoint_cam,
                    image,
                    view_target_entry,
                )

            last_metrics = metrics
            progress_bar.set_postfix(
                {
                    "loss": f"{metrics['loss_total']:.5f}",
                    "l1": f"{metrics['loss_l1']:.5f}",
                    "ssim": f"{metrics['ssim']:.4f}",
                    "sup": f"{metrics['supervision_mean']:.3f}",
                    "target": target_mode,
                    "train": train_count,
                }
            )

            if (
                iteration == int(opt.iterations)
                and bool(getattr(args, "opacity_recalibration", False))
                and not bool(opacity_recalibration_metadata.get("enabled", False))
            ):
                print("\n[MOD_COB_INPAINT][OPACITY] Starting opacity recalibration")
                opacity_recalibration_metadata = run_opacity_recalibration(
                    scene=scene,
                    gaussians=scene.gaussians,
                    dataset=dataset,
                    opt=opt,
                    pipe=pipe,
                    args=args,
                    desk_object_id=desk_object_id,
                    decouple_object_ids=decouple_object_ids,
                    background=background,
                )
                print(
                    "[MOD_COB_INPAINT][OPACITY] "
                    f"objects={opacity_recalibration_metadata.get('object_id', [])} "
                    f"mean_opacity={opacity_recalibration_metadata.get('mean_opacity_before', 0.0):.4f}"
                    f"->{opacity_recalibration_metadata.get('mean_opacity_after', 0.0):.4f}"
                )

            metadata = _metadata_payload(
                iteration=iteration,
                source_iteration=source_iteration,
                desk_object_id=desk_object_id,
                decouple_object_ids=decouple_object_ids,
                excluded_object_ids=excluded_object_ids,
                inpaint_state=inpaint_state,
                desk_atlas_state=desk_atlas_state,
                support_stats=merged_support_stats,
                init_footprint_stats=init_footprint_stats,
                train_count=train_count,
                frozen_count=frozen_count,
                opacity_recalibration=opacity_recalibration_metadata,
                removal_gt={
                    "enabled": bool(removal_gt_enabled),
                    "ratio": float(removal_gt_ratio),
                    "schedule": {
                        "removal_steps": int(removal_gt_schedule.numerator),
                        "period": int(removal_gt_schedule.denominator),
                    },
                    "directory": removal_gt_dir,
                    "used_steps": int(removal_gt_steps),
                    "constructed_steps": int(constructed_gt_steps),
                },
            )
            save_interval = int(args.save_interval)
            checkpoint_interval = int(args.checkpoint_interval)
            should_save_snapshot = iteration == int(opt.iterations) or (save_interval > 0 and iteration % save_interval == 0)
            should_save_checkpoint = iteration == int(opt.iterations) or (
                checkpoint_interval > 0 and iteration % checkpoint_interval == 0
            )
            if should_save_snapshot or should_save_checkpoint:
                _save_inpaint_outputs(
                    dataset.model_path,
                    iteration,
                    scene.gaussians,
                    metadata,
                    save_point_cloud=should_save_snapshot,
                    save_checkpoint=should_save_checkpoint,
                )

    return {
        "scene": scene,
        "desk_atlas_state": desk_atlas_state,
        "inpaint_state": inpaint_state,
        "training_state": training_state,
        "metrics": last_metrics,
    }


def _add_inpaint_args(parser: ArgumentParser) -> None:
    parser.add_argument("--source_iteration", default=30000, type=int)
    parser.add_argument("--segmentation_checkpoint", default=None, type=str)
    parser.add_argument("--desk_object_id", default=None)
    parser.add_argument("--decouple_object_id", nargs="+", default=None)
    parser.add_argument("--desk_atlas_dir", type=str, default="desk_atlas")
    parser.add_argument("--completed_texture_path", type=str, default=None)
    parser.add_argument("--removal_GT", action="store_true", default=False)
    parser.add_argument("--removal_GT_ratio", type=float, default=0.75)
    parser.add_argument("--hole_init_stride_px", type=int, default=4)
    parser.add_argument("--view_supervision_boundary_px", type=int, default=16)
    parser.add_argument("--desk_support_shrink_px", type=float, default=15.0)
    parser.add_argument("--desk_support_close_kernel_size", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--checkpoint_interval", type=int, default=0)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--save_training_vis", action="store_true", default=False)
    parser.add_argument("--save_training_vis_iteration", type=int, default=1000)
    parser.add_argument("--opacity_recalibration", action="store_true", default=False)
    parser.add_argument("--opacity_recalibration_iterations", type=int, default=500)
    parser.add_argument("--opacity_recalibration_lr", type=float, default=0.005)
    parser.add_argument("--opacity_recalibration_dilation_px", type=int, default=3)
    parser.add_argument("--opacity_recalibration_reg_weight", type=float, default=0.01)
    parser.add_argument("--opacity_recalibration_object_id", nargs="+", default=None)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true", default=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="mod-COB-GS RGB desk-atlas inpaint training script")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.set_defaults(iterations=35000)
    _add_inpaint_args(parser)
    args = get_combined_args(parser)

    safe_state(args.quiet)
    random_seed = int(getattr(args, "seed", 0) or 0)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Optimizing " + args.model_path)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args,
    )
    print("\nRGB inpaint training complete.")
