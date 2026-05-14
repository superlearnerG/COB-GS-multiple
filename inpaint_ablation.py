#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from random import randint
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
DEFAULT_SUPPRESSED_OPACITY = 1e-6
FULL_GAUSSIAN_STATE_KEYS = {
    "active_sh_degree",
    "xyz",
    "features_dc",
    "features_rest",
    "scaling",
    "rotation",
    "opacity",
    "max_radii2D",
    "xyz_gradient_accum",
    "denom",
    "spatial_lr_scale",
}


def _parse_object_ids(raw_value) -> List[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        object_ids: List[int] = []
        for item in raw_value:
            object_ids.extend(_parse_object_ids(item))
        return object_ids
    raw_text = str(raw_value).replace(",", " ").strip()
    if not raw_text:
        return []
    object_ids = []
    for item in raw_text.split():
        object_id = int(item)
        if object_id <= 0:
            raise ValueError(f"Object ids must be positive, got {object_id}.")
        object_ids.append(object_id)
    return object_ids


def _dedupe_preserve_ids(raw_value) -> List[int]:
    object_ids = []
    seen = set()
    for object_id in _parse_object_ids(raw_value):
        if object_id in seen:
            continue
        object_ids.append(object_id)
        seen.add(object_id)
    if not object_ids:
        raise ValueError("--preserve_object_id must contain at least one positive object id.")
    return object_ids


def _dedupe_object_ids(raw_value) -> List[int]:
    object_ids = []
    seen = set()
    for object_id in _parse_object_ids(raw_value):
        if object_id in seen:
            continue
        object_ids.append(object_id)
        seen.add(object_id)
    return object_ids


def _unpack_gaussian_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], Optional[int]]:
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
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a Gaussian state dict.")
    return model_state, marker


def _has_full_gaussian_state(model_state: Dict[str, Any]) -> bool:
    return FULL_GAUSSIAN_STATE_KEYS.issubset(model_state.keys())


def _state_without_optimizer(model_state: Dict[str, Any]) -> Dict[str, Any]:
    fresh_state = dict(model_state)
    fresh_state["optimizer"] = None
    return fresh_state


def _resolve_segmentation_checkpoint(model_path: str, checkpoint_path: Optional[str]) -> str:
    if checkpoint_path:
        return os.path.abspath(checkpoint_path)
    return os.path.abspath(os.path.join(model_path, "multi_object", "final_multi_object.pth"))


def _resolve_base_checkpoint(model_path: str, checkpoint_path: Optional[str], source_iteration: int) -> str:
    if checkpoint_path:
        return os.path.abspath(checkpoint_path)
    return os.path.abspath(os.path.join(model_path, f"chkpnt{source_iteration}.pth"))


def _require_file(path: str, desc: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def _ensure_mask_state(gaussians: GaussianModel) -> None:
    n_points = int(gaussians.get_xyz.shape[0])
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    if getattr(gaussians, "_mask", None) is None or int(gaussians._mask.numel()) != n_points:
        gaussians._mask = nn.Parameter(torch.zeros((n_points,), dtype=dtype, device=device), requires_grad=False)
    else:
        gaussians._mask = nn.Parameter(gaussians._mask.detach().to(device=device, dtype=dtype), requires_grad=False)


def _ensure_exposure_state(gaussians: GaussianModel, train_cameras: Sequence[Any]) -> None:
    image_names = [camera.image_name for camera in train_cameras]
    if not hasattr(gaussians, "exposure_mapping") or not isinstance(getattr(gaussians, "exposure_mapping"), dict):
        gaussians.exposure_mapping = {name: idx for idx, name in enumerate(image_names)}
    if not hasattr(gaussians, "pretrained_exposures"):
        gaussians.pretrained_exposures = None
    if not hasattr(gaussians, "_exposure") or not torch.is_tensor(getattr(gaussians, "_exposure", None)):
        exposure = torch.eye(3, 4, device=gaussians.get_xyz.device)[None].repeat(len(image_names), 1, 1)
        gaussians._exposure = nn.Parameter(exposure.requires_grad_(True))


def _restore_full_state_for_ablation(
    gaussians: GaussianModel,
    model_state: Dict[str, Any],
    opt: OptimizationParams,
    train_cameras: Sequence[Any],
) -> None:
    xyz_gradient_accum, denom, _ = gaussians._restore_from_state_dict(_state_without_optimizer(model_state))
    _ensure_exposure_state(gaussians, train_cameras)
    _ensure_mask_state(gaussians)
    gaussians.training_setup(opt)
    gaussians.xyz_gradient_accum = xyz_gradient_accum
    gaussians.denom = denom
    _ensure_mask_state(gaussians)


def _apply_object_state(gaussians: GaussianModel, model_state: Dict[str, Any], checkpoint_path: str) -> None:
    object_id = model_state["object_id"].detach().reshape(-1).to(
        device=gaussians.get_xyz.device,
        dtype=torch.int32,
    )
    num_gaussians = int(gaussians.get_xyz.shape[0])
    if int(object_id.numel()) != num_gaussians:
        raise RuntimeError(
            f"Object label count from {checkpoint_path} ({int(object_id.numel())}) does not match "
            f"the base Gaussian count ({num_gaussians}). Pass a full final_multi_object checkpoint "
            "or a matching --base_checkpoint."
        )
    object_score = model_state.get("object_score")
    if object_score is None or int(object_score.numel()) != num_gaussians:
        object_score = torch.zeros((num_gaussians,), dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
    else:
        object_score = object_score.detach().reshape(-1).to(
            device=gaussians.get_xyz.device,
            dtype=gaussians.get_xyz.dtype,
        )
    gaussians.object_id = object_id
    gaussians.object_score = object_score
    _ensure_mask_state(gaussians)


def _load_ablation_start_state(
    scene: Scene,
    opt: OptimizationParams,
    segmentation_checkpoint: str,
    base_checkpoint: str,
) -> Tuple[Optional[int], str]:
    _require_file(segmentation_checkpoint, "segmentation checkpoint")
    model_state, marker = _unpack_gaussian_checkpoint(segmentation_checkpoint)
    if "object_id" not in model_state:
        raise ValueError(f"Segmentation checkpoint {segmentation_checkpoint} does not contain object_id.")

    if _has_full_gaussian_state(model_state):
        _restore_full_state_for_ablation(scene.gaussians, model_state, opt, scene.getTrainCameras())
        return marker, "segmentation_checkpoint"

    _require_file(base_checkpoint, "base checkpoint")
    base_state, _ = _unpack_gaussian_checkpoint(base_checkpoint)
    if not _has_full_gaussian_state(base_state):
        raise ValueError(f"Base checkpoint {base_checkpoint} does not contain a full Gaussian state.")
    _restore_full_state_for_ablation(scene.gaussians, base_state, opt, scene.getTrainCameras())
    _apply_object_state(scene.gaussians, model_state, segmentation_checkpoint)
    return marker, "base_checkpoint_plus_object_labels"


def _assigned_positive_labels(gaussians: GaussianModel) -> List[int]:
    labels = torch.unique(gaussians.get_object_id.detach())
    return sorted(int(label.item()) for label in labels if int(label.item()) > 0)


def _set_opacity_for_mask(gaussians: GaussianModel, opacity_mask: torch.Tensor, opacity_value: float) -> None:
    if not (0.0 < float(opacity_value) < 1.0):
        raise ValueError(f"--suppressed_opacity must be in (0, 1), got {opacity_value}.")
    if int(opacity_mask.numel()) != int(gaussians.get_xyz.shape[0]):
        raise RuntimeError(
            f"Opacity suppression mask has {int(opacity_mask.numel())} entries, "
            f"but the Gaussian model has {int(gaussians.get_xyz.shape[0])} points."
        )
    if not bool(opacity_mask.any().item()):
        return

    target_opacity = torch.full_like(gaussians._opacity[opacity_mask], float(opacity_value))
    target_logit = gaussians.inverse_opacity_activation(target_opacity)
    with torch.no_grad():
        gaussians._opacity[opacity_mask] = target_logit


def _initialize_non_preserved_opacity(
    gaussians: GaussianModel,
    preserve_object_ids: Sequence[int],
    suppressed_opacity: float,
) -> Dict[str, Any]:
    object_id = gaussians.get_object_id
    assigned_labels = set(_assigned_positive_labels(gaussians))
    missing = [object_id_value for object_id_value in preserve_object_ids if object_id_value not in assigned_labels]
    if missing:
        raise ValueError(
            f"--preserve_object_id requested ids {missing}, but checkpoint object ids are "
            f"{sorted(assigned_labels)}."
        )

    preserve_filter = object_id == 0
    for object_id_value in preserve_object_ids:
        preserve_filter = torch.logical_or(preserve_filter, object_id == int(object_id_value))
    initial_count = int(object_id.shape[0])
    preserve_count = int(preserve_filter.sum().item())
    if preserve_count <= 0:
        raise RuntimeError("Preserve filter selected no Gaussians.")

    initialized_filter = torch.logical_not(preserve_filter).detach()
    _set_opacity_for_mask(gaussians, initialized_filter, suppressed_opacity)
    _ensure_mask_state(gaussians)
    return {
        "initial_gaussians": initial_count,
        "preserved_gaussians": preserve_count,
        "opacity_initialized_gaussians": initial_count - preserve_count,
        "preserve_object_id": [int(v) for v in preserve_object_ids],
        "always_preserved_background_id": 0,
        "initial_opacity": float(suppressed_opacity),
        "mode": "initial_opacity_only",
    }


def _resolve_id_in_background(
    gaussians: GaussianModel,
    raw_value,
    visible_object_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    requested_ids = _dedupe_object_ids(raw_value)
    if not requested_ids:
        return []
    assigned_labels = set(_assigned_positive_labels(gaussians)) if visible_object_ids is None else set(int(v) for v in visible_object_ids)
    missing = [object_id for object_id in requested_ids if object_id not in assigned_labels]
    if missing:
        raise ValueError(
            f"--id_in_background requested ids {missing}, but these ids are not visible in "
            f"the ablation scene. Add them to --preserve_object_id first."
        )
    return requested_ids


def _view_output_filename(view) -> str:
    image_name = os.path.basename(str(getattr(view, "image_name", "")).strip())
    if not image_name:
        raise ValueError("Cannot save render because view.image_name is empty.")
    if not os.path.splitext(image_name)[1]:
        image_name = f"{image_name}.png"
    return image_name


def _view_output_filenames(views: Sequence[Any], split_name: str) -> List[str]:
    output_names = [_view_output_filename(view) for view in views]
    duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Cannot save {split_name} renders with original filenames because duplicate basenames exist: "
            f"{duplicates}"
        )
    return output_names


def _resolve_image_path(root_dir: Path, image_name: str, desc: str) -> Path:
    basename = os.path.basename(str(image_name))
    stem, ext = os.path.splitext(basename)
    candidates = []
    exact_path = root_dir / basename
    if basename and exact_path.exists():
        candidates.append(exact_path)

    if ext:
        for suffix in IMAGE_EXTENSIONS:
            variant = root_dir / f"{stem}{suffix}"
            if variant.exists() and variant not in candidates:
                candidates.append(variant)
    else:
        for suffix in IMAGE_EXTENSIONS:
            variant = root_dir / f"{stem}{suffix}"
            if variant.exists() and variant not in candidates:
                candidates.append(variant)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous {desc} image for '{image_name}' under {root_dir}: {candidates}")
    raise FileNotFoundError(f"{desc} image not found for '{image_name}' under {root_dir}")


def _load_supervision_tensor(image_path: Path, camera) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        expected_size = (int(camera.image_width), int(camera.image_height))
        if image.size != expected_size:
            raise RuntimeError(
                f"Supervision image resolution mismatch for '{camera.image_name}' at {image_path}: "
                f"expected {expected_size}, got {image.size}."
            )
        image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


def build_supervision_cache(supervision_path: str, cameras: Sequence[Any]) -> Tuple[Dict[int, torch.Tensor], Dict[str, str]]:
    root = Path(supervision_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--supervision_path directory not found: {root}")
    if not any(path.is_file() and path.suffix in IMAGE_EXTENSIONS for path in root.iterdir()):
        raise FileNotFoundError(f"--supervision_path contains no supported images: {root}")

    cache: Dict[int, torch.Tensor] = {}
    matched_paths: Dict[str, str] = {}
    for camera in cameras:
        image_path = _resolve_image_path(root, camera.image_name, "supervision")
        cache[int(camera.uid)] = _load_supervision_tensor(image_path, camera)
        matched_paths[str(camera.image_name)] = str(image_path)
    return cache, matched_paths


def compute_ablation_loss(image: torch.Tensor, target: torch.Tensor, opt: OptimizationParams) -> Tuple[torch.Tensor, Dict[str, float]]:
    ll1 = l1_loss(image, target)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
    else:
        ssim_value = ssim(image, target)
    loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    return loss, {
        "l1": float(ll1.item()),
        "ssim": float(ssim_value.item()),
        "loss": float(loss.item()),
    }


def save_ablation_outputs(
    model_path: str,
    gaussians: GaussianModel,
    iteration: int,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    ablation_root = os.path.join(model_path, "ablation")
    point_cloud_dir = os.path.join(ablation_root, "point_cloud", f"iteration_{iteration}")
    os.makedirs(point_cloud_dir, exist_ok=True)
    os.makedirs(ablation_root, exist_ok=True)

    ply_path = os.path.join(point_cloud_dir, "point_cloud.ply")
    point_cloud_pth = os.path.join(point_cloud_dir, "point_cloud.pth")
    checkpoint_path = os.path.join(ablation_root, f"chkpnt{iteration}.pth")
    final_path = os.path.join(ablation_root, "final_ablation.pth")
    metadata_path = os.path.join(ablation_root, "metadata.json")

    gaussians.save_ply(ply_path)
    captured = gaussians.capture(include_mask=True)
    torch.save((captured, int(iteration)), checkpoint_path)
    torch.save({"gaussians": captured, "iteration": int(iteration), "metadata": metadata}, final_path)
    torch.save({"gaussians": captured, "iteration": int(iteration), "metadata": metadata}, point_cloud_pth)
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    return {
        "ablation_root": ablation_root,
        "point_cloud_ply": ply_path,
        "point_cloud_pth": point_cloud_pth,
        "checkpoint": checkpoint_path,
        "final_checkpoint": final_path,
        "metadata": metadata_path,
    }


def _render_target(
    camera,
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
    opt: OptimizationParams,
    train_test_exp: bool,
    gaussian_filter: Optional[torch.Tensor],
) -> torch.Tensor:
    selected_count = int(gaussians.get_xyz.shape[0]) if gaussian_filter is None else int(gaussian_filter.sum().item())
    if selected_count == 0:
        return background[:, None, None].expand(3, int(camera.image_height), int(camera.image_width))
    renders = render(
        camera,
        gaussians,
        pipe,
        background,
        opt,
        use_trained_exp=train_test_exp,
        gaussian_filter=gaussian_filter,
    )
    return renders["render"]


def render_ablation_split(
    render_root: str,
    split_name: str,
    views: Sequence[Any],
    gaussians: GaussianModel,
    pipe: PipelineParams,
    background: torch.Tensor,
    train_test_exp: bool,
    opt: OptimizationParams,
    preserve_object_ids: Sequence[int],
    desk_object_id: Optional[int],
    id_in_background: Sequence[int],
    only_desk_background: bool,
) -> None:
    visible_label_set = set(int(v) for v in preserve_object_ids)
    labels = [label for label in _assigned_positive_labels(gaussians) if int(label) in visible_label_set]
    output_names = _view_output_filenames(views, split_name)
    render_targets: List[Tuple[str, Optional[torch.Tensor]]] = []

    if not only_desk_background:
        render_targets.append(("original", None))
        for label in labels:
            if desk_object_id is not None and int(label) == int(desk_object_id):
                continue
            render_targets.append((f"object_{label}", gaussians.get_object_filter(label)))

    desk_background_filter = gaussians.get_object_filter(0)
    if desk_object_id is not None and int(desk_object_id) in labels:
        desk_background_filter = torch.logical_or(desk_background_filter, gaussians.get_object_filter(int(desk_object_id)))
    for object_id in id_in_background:
        desk_background_filter = torch.logical_or(desk_background_filter, gaussians.get_object_filter(int(object_id)))
    desk_background_name = "desk+background"
    if id_in_background:
        desk_background_name += "_" + "_".join(str(int(object_id)) for object_id in id_in_background)
    render_targets.append((desk_background_name, desk_background_filter))

    with torch.no_grad():
        for target_name, gaussian_filter in render_targets:
            image_root = os.path.join(render_root, target_name, split_name, "render")
            os.makedirs(image_root, exist_ok=True)
            for idx, view in enumerate(tqdm(views, desc=f"Rendering {target_name} {split_name}")):
                render_image = _render_target(
                    view,
                    gaussians,
                    pipe,
                    background,
                    opt,
                    train_test_exp,
                    gaussian_filter,
                )
                torchvision.utils.save_image(render_image, os.path.join(image_root, output_names[idx]))


def render_ablation_outputs(
    dataset: ModelParams,
    pipe: PipelineParams,
    opt: OptimizationParams,
    scene: Scene,
    preserve_object_ids: Sequence[int],
    desk_object_id: Optional[int],
    id_in_background: Sequence[int],
    only_desk_background: bool,
    skip_train: bool,
    skip_test: bool,
) -> str:
    render_root = os.path.join(dataset.model_path, "ablation", "decouple+inpaint")
    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    if not skip_train:
        render_ablation_split(
            render_root,
            "train",
            scene.getTrainCameras(),
            scene.gaussians,
            pipe,
            background,
            dataset.train_test_exp,
            opt,
            preserve_object_ids,
            desk_object_id,
            id_in_background,
            only_desk_background,
        )
    if not skip_test:
        render_ablation_split(
            render_root,
            "test",
            scene.getTestCameras(),
            scene.gaussians,
            pipe,
            background,
            dataset.train_test_exp,
            opt,
            preserve_object_ids,
            desk_object_id,
            id_in_background,
            only_desk_background,
        )
    return render_root


def train_ablation(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, args) -> Dict[str, Any]:
    source_iteration = int(args.source_iteration)
    if source_iteration <= 0:
        raise ValueError(f"--source_iteration must be positive, got {source_iteration}.")
    if int(opt.iterations) <= source_iteration:
        raise ValueError(f"--iterations ({opt.iterations}) must be > --source_iteration ({source_iteration}).")

    preserve_object_ids = _dedupe_preserve_ids(getattr(args, "preserve_object_id", None))
    opt.include_mask = False
    opt.inpainting = True
    opt.random_background = False

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=source_iteration, shuffle=False)
    segmentation_checkpoint = _resolve_segmentation_checkpoint(dataset.model_path, getattr(args, "segmentation_checkpoint", None))
    base_checkpoint = _resolve_base_checkpoint(dataset.model_path, getattr(args, "base_checkpoint", None), source_iteration)
    suppressed_opacity = float(getattr(args, "suppressed_opacity", DEFAULT_SUPPRESSED_OPACITY))
    checkpoint_marker, restore_mode = _load_ablation_start_state(
        scene,
        opt,
        segmentation_checkpoint,
        base_checkpoint,
    )
    opacity_initialization_stats = _initialize_non_preserved_opacity(
        scene.gaussians,
        preserve_object_ids,
        suppressed_opacity,
    )
    id_in_background = _resolve_id_in_background(scene.gaussians, getattr(args, "id_in_background", None), preserve_object_ids)

    supervision_cache, supervision_matches = build_supervision_cache(args.supervision_path, scene.getTrainCameras())

    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss = 0.0
    last_metrics: Dict[str, float] = {}

    progress_bar = tqdm(range(source_iteration + 1, int(opt.iterations) + 1), desc="AblationInpaint")
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
        render_pkg = render(
            viewpoint_cam,
            scene.gaussians,
            pipe,
            background,
            opt,
            use_trained_exp=dataset.train_test_exp,
        )
        image = render_pkg["render"]
        if getattr(viewpoint_cam, "alpha_mask", None) is not None:
            image = image * viewpoint_cam.alpha_mask.to(device=image.device)

        target = supervision_cache[int(viewpoint_cam.uid)].to(device=image.device)
        loss, metrics = compute_ablation_loss(image, target, opt)
        loss.backward()

        with torch.no_grad():
            scene.gaussians.optimizer.step()
            scene.gaussians.optimizer.zero_grad(set_to_none=True)
            ema_loss = 0.4 * float(metrics["loss"]) + 0.6 * ema_loss
            last_metrics = metrics
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{ema_loss:.6f}",
                        "l1": f"{metrics['l1']:.6f}",
                        "ssim": f"{metrics['ssim']:.4f}",
                    }
                )

    metadata = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "source_iteration": int(source_iteration),
        "final_iteration": int(opt.iterations),
        "segmentation_checkpoint": segmentation_checkpoint,
        "base_checkpoint": base_checkpoint if restore_mode == "base_checkpoint_plus_object_labels" else None,
        "checkpoint_marker": checkpoint_marker,
        "restore_mode": restore_mode,
        "supervision_path": os.path.abspath(args.supervision_path),
        "supervision_matches": supervision_matches,
        "opacity_initialization": opacity_initialization_stats,
        "render": {
            "id_in_background": [int(v) for v in id_in_background],
            "only_desk_background": bool(getattr(args, "only_desk_background", False)),
        },
        "last_metrics": last_metrics,
        "lambda_dssim": float(opt.lambda_dssim),
    }
    output_paths = save_ablation_outputs(dataset.model_path, scene.gaussians, int(opt.iterations), metadata)
    metadata["outputs"] = output_paths
    with open(output_paths["metadata"], "w") as file:
        json.dump(metadata, file, indent=2)

    if not bool(getattr(args, "skip_render", False)):
        desk_object_id = int(args.desk_object_id) if getattr(args, "desk_object_id", None) is not None else None
        render_root = render_ablation_outputs(
            dataset,
            pipe,
            opt,
            scene,
            preserve_object_ids,
            desk_object_id,
            id_in_background,
            bool(getattr(args, "only_desk_background", False)),
            bool(getattr(args, "skip_train_render", False)),
            bool(getattr(args, "skip_test_render", False)),
        )
        metadata["outputs"]["render_root"] = render_root
        with open(output_paths["metadata"], "w") as file:
            json.dump(metadata, file, indent=2)

    return metadata


def _add_ablation_args(parser: ArgumentParser) -> None:
    parser.add_argument("--source_iteration", default=30000, type=int)
    parser.add_argument("--segmentation_checkpoint", default=None, type=str)
    parser.add_argument("--base_checkpoint", default=None, type=str)
    parser.add_argument("--preserve_object_id", nargs="+", default=None)
    parser.add_argument("--suppressed_opacity", default=DEFAULT_SUPPRESSED_OPACITY, type=float)
    parser.add_argument("--desk_object_id", default=255, type=int)
    parser.add_argument("--id_in_background", nargs="+", default=None)
    parser.add_argument("--only_desk+background", "--only_desk_background", dest="only_desk_background", action="store_true", default=False)
    parser.add_argument("--supervision_path", default=None, type=str)
    parser.add_argument("--skip_render", action="store_true", default=False)
    parser.add_argument("--skip_train_render", action="store_true", default=False)
    parser.add_argument("--skip_test_render", action="store_true", default=False)
    parser.add_argument("--debug_from", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", default=0, type=int)


if __name__ == "__main__":
    parser = ArgumentParser(description="mod-COB-GS ablation inpaint training script")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.set_defaults(iterations=35000)
    _add_ablation_args(parser)
    args = get_combined_args(parser)
    if args.supervision_path is None:
        parser.error("--supervision_path is required")

    safe_state(args.quiet)
    random_seed = int(getattr(args, "seed", 0) or 0)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Running ablation inpaint for " + args.model_path)
    train_ablation(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args,
    )
    print("\nAblation inpaint complete.")


# cd /home/kunxinguang/Sourcecode/3D-Seg/GROUPING/siga26/mod-COB-GS

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint_ablation.py \
#   -s ../data/bedroom \
#   -m ../output/bedroom/cobgs/5010000 \
#   --source_iteration 30000 \
#   --iterations 35000 \
#   --segmentation_checkpoint ../output/bedroom/cobgs/5010000/multi_object/final_multi_object.pth \
#   --preserve_object_id 255 17 34 51 68 \
#   --supervision_path /path/to/inpaint_supervision \
#   --eval

# 注意：如果你希望 desk 也保留，需要把 desk id 例如 255 放进 --preserve_object_id。
# --desk_object_id 255 只影响 render 分组命名，不会自动保留该 id。
