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

import copy
import json
import os
from argparse import ArgumentParser

import mediapy as media
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def infer_scene_type(source_path):
    lower_path = source_path.lower()
    surrounding_tokens = ("360", "mip", "tandt", "tank", "tnt", "garden", "kitchen", "truck")
    if any(token in lower_path for token in surrounding_tokens):
        return "surrounding"
    return "forward"


def resolve_segmentation_n4views(dataset, requested_n4views):
    scene_type = infer_scene_type(dataset.source_path)
    if requested_n4views is not None:
        return requested_n4views, scene_type
    return (14 if scene_type == "surrounding" else 22), scene_type


def _unpack_gaussian_checkpoint(checkpoint_path):
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


def _resolve_multi_object_checkpoint(model_path, loaded_iteration, args):
    explicit_checkpoint = getattr(args, "segmentation_checkpoint", None)
    if explicit_checkpoint:
        checkpoint_path = os.path.abspath(explicit_checkpoint)
        model_state, marker = _unpack_gaussian_checkpoint(checkpoint_path)
        if "object_id" not in model_state:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain object_id.")
        return checkpoint_path, model_state, marker

    candidates = []
    if loaded_iteration:
        candidates.extend(
            [
                os.path.join(model_path, "point_cloud", f"iteration_{loaded_iteration}", "point_cloud.pth"),
                os.path.join(model_path, f"chkpnt{loaded_iteration}.pth"),
            ]
        )
    candidates.append(os.path.join(model_path, "multi_object", "final_multi_object.pth"))

    skipped_without_ids = []
    for checkpoint_path in candidates:
        if not os.path.exists(checkpoint_path):
            continue
        try:
            model_state, marker = _unpack_gaussian_checkpoint(checkpoint_path)
        except ValueError:
            skipped_without_ids.append(checkpoint_path)
            continue
        if "object_id" not in model_state:
            skipped_without_ids.append(checkpoint_path)
            continue
        return checkpoint_path, model_state, marker

    detail = ""
    if skipped_without_ids:
        detail = " Existing checkpoints without object_id: " + ", ".join(skipped_without_ids)
    raise FileNotFoundError(
        f"Could not find a multi-object checkpoint with object_id under {model_path} "
        f"for iteration {loaded_iteration}.{detail}"
    )


def _apply_object_labels_from_state(gaussians, model_state, checkpoint_path):
    object_id = model_state["object_id"].detach().reshape(-1).to(
        device=gaussians.get_xyz.device,
        dtype=torch.int32,
    )
    num_gaussians = int(gaussians.get_xyz.shape[0])
    if int(object_id.numel()) != num_gaussians:
        raise RuntimeError(
            f"Object label count from {checkpoint_path} ({int(object_id.numel())}) does not match "
            f"the loaded iteration Gaussian count ({num_gaussians}). Use a matching --iteration "
            f"or pass a matching --segmentation_checkpoint."
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
    if gaussians._mask is None or int(gaussians._mask.numel()) != num_gaussians:
        gaussians._mask = torch.zeros((num_gaussians,), dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)


def _has_full_gaussian_state(model_state):
    required_keys = {
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
    return required_keys.issubset(model_state.keys())


def _restore_or_apply_multi_object_state(gaussians, model_state, checkpoint_path, args):
    object_count = int(model_state["object_id"].detach().reshape(-1).numel())
    loaded_count = int(gaussians.get_xyz.shape[0])
    if object_count != loaded_count:
        if not _has_full_gaussian_state(model_state):
            raise RuntimeError(
                f"Object label count from {checkpoint_path} ({object_count}) does not match "
                f"the loaded iteration Gaussian count ({loaded_count}), and the checkpoint does "
                "not contain a full Gaussian state to restore."
            )
        print(
            f"Restoring full Gaussian state from {checkpoint_path} because object label count "
            f"({object_count}) does not match loaded iteration count ({loaded_count})."
        )
        gaussians.restore(model_state, args, mode="test")
        restored_count = int(gaussians.get_xyz.shape[0])
        if restored_count != object_count:
            raise RuntimeError(
                f"Restored Gaussian count from {checkpoint_path} ({restored_count}) still does "
                f"not match object label count ({object_count})."
            )
        return

    _apply_object_labels_from_state(gaussians, model_state, checkpoint_path)


def _infer_desk_object_id(labels, requested_desk_object_id=None):
    positive_labels = sorted({int(label) for label in labels if int(label) > 0})
    if not positive_labels:
        raise RuntimeError("No object_id > 0 was found; cannot infer desk_object_id.")

    if requested_desk_object_id is not None:
        desk_object_id = int(requested_desk_object_id)
        if desk_object_id not in positive_labels:
            raise ValueError(f"desk_object_id {desk_object_id} is not present in checkpoint object ids {positive_labels}.")
        return desk_object_id

    return 255 if 255 in positive_labels else max(positive_labels)


def _parse_id_list(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        ids = []
        for item in raw_value:
            ids.extend(_parse_id_list(item))
        return ids
    raw_value = str(raw_value).replace(",", " ").strip()
    if not raw_value:
        return []
    parsed_ids = []
    for item in raw_value.split():
        object_id = int(item)
        if object_id <= 0:
            raise ValueError(f"Object ids must be positive, got {object_id}.")
        parsed_ids.append(object_id)
    return parsed_ids


def _resolve_id_in_background(args, assigned_labels):
    requested_ids = []
    seen_ids = set()
    for object_id in _parse_id_list(getattr(args, "id_in_background", None)):
        if object_id in seen_ids:
            continue
        requested_ids.append(object_id)
        seen_ids.add(object_id)
    assigned_set = set(int(label) for label in assigned_labels)
    missing = [object_id for object_id in requested_ids if object_id not in assigned_set]
    if missing:
        raise ValueError(
            f"--id_in_background requested ids {missing}, but checkpoint object ids are "
            f"{sorted(assigned_set)}."
        )
    return requested_ids


def _view_output_filename(view):
    image_name = os.path.basename(str(getattr(view, "image_name", "")).strip())
    if not image_name:
        raise ValueError("Cannot save render with original filename because view.image_name is empty.")
    if not os.path.splitext(image_name)[1]:
        image_name = f"{image_name}.png"
    return image_name


def _view_output_filenames(views, split_name):
    output_names = [_view_output_filename(view) for view in views]
    duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Cannot save {split_name} renders with original filenames because duplicate basenames exist: "
            f"{duplicates}"
        )
    return output_names


def _sequence_output_filenames(views):
    return [f"{idx:05d}.png" for idx, _ in enumerate(views)]


def _normalize_np(x):
    return x / np.linalg.norm(x)


def _pad_poses(poses):
    bottom = np.broadcast_to([0, 0, 0, 1.0], poses[..., :1, :4].shape)
    return np.concatenate([poses[..., :3, :4], bottom], axis=-2)


def _unpad_poses(poses):
    return poses[..., :3, :4]


def _viewmatrix(lookdir, up, position):
    vec2 = _normalize_np(lookdir)
    vec0 = _normalize_np(np.cross(up, vec2))
    vec1 = _normalize_np(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def _focus_point_fn(poses):
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


def _transform_poses_pca(poses):
    translations = poses[:, :3, 3]
    translation_mean = translations.mean(axis=0)
    centered = translations - translation_mean
    eigval, eigvec = np.linalg.eig(centered.T @ centered)
    inds = np.argsort(eigval)[::-1]
    rot = eigvec[:, inds].T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -translation_mean[:, None]], axis=-1)
    poses_recentered = _unpad_poses(transform @ _pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform
    return poses_recentered, transform


def _generate_ellipse_path(poses, n_frames):
    center = _focus_point_fn(poses)
    offset = np.array([center[0], center[1], 0])
    scale = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -scale + offset
    high = scale + offset

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)[:-1]
    positions = np.stack(
        [
            low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
            low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
            np.zeros_like(theta),
        ],
        axis=-1,
    )

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    return np.stack([_viewmatrix(position - center, up, position) for position in positions])


def generate_render_path(viewpoint_cameras, n_frames):
    if not viewpoint_cameras:
        raise ValueError("Cannot generate render path without source cameras.")
    if n_frames <= 0:
        raise ValueError("--render_path_frames must be positive.")

    c2ws = np.array(
        [
            np.linalg.inv(np.asarray((cam.world_view_transform.T).detach().cpu().numpy()))
            for cam in viewpoint_cameras
        ]
    )
    poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    poses_recentered, colmap_to_world_transform = _transform_poses_pca(poses)
    new_poses = _generate_ellipse_path(poses_recentered, n_frames=n_frames)
    new_poses = np.linalg.inv(colmap_to_world_transform) @ _pad_poses(new_poses)

    trajectory = []
    for idx, c2w in enumerate(new_poses):
        c2w = c2w @ np.diag([1, -1, -1, 1])
        cam = copy.deepcopy(viewpoint_cameras[0])
        cam.image_name = f"{idx:05d}.png"
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        trajectory.append(cam)
    return trajectory


def scene_render_path_cameras(scene):
    cameras = list(scene.getTrainCameras())
    cameras.extend(scene.getTestCameras())
    return cameras


def _read_even_rgb_frame(path):
    frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    height = frame.shape[0] - (frame.shape[0] % 2)
    width = frame.shape[1] - (frame.shape[1] % 2)
    return frame[:height, :width]


def write_mp4_from_png_sequence(frame_paths, output_path, fps):
    existing_paths = [path for path in frame_paths if os.path.exists(path)]
    if not existing_paths:
        print(f"No frames found for video {output_path}; skipping.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    first_frame = _read_even_rgb_frame(existing_paths[0])
    video_kwargs = {
        "shape": first_frame.shape[:2],
        "codec": "h264",
        "fps": fps,
        "crf": 18,
    }
    print(f"Making video {output_path}...")
    with media.VideoWriter(output_path, **video_kwargs, input_format="rgb") as writer:
        for frame_path in tqdm(existing_paths, desc=f"Writing {os.path.basename(output_path)}"):
            frame = _read_even_rgb_frame(frame_path)
            if frame.shape[:2] != first_frame.shape[:2]:
                frame = frame[: first_frame.shape[0], : first_frame.shape[1]]
            writer.add_image(frame)


def write_traj_videos(output_root, frame_names, fps, include_depth=False, render_subdir="renders"):
    render_paths = [os.path.join(output_root, render_subdir, name) for name in frame_names]
    write_mp4_from_png_sequence(render_paths, os.path.join(output_root, "render_traj_color.mp4"), fps)
    if include_depth:
        depth_paths = [os.path.join(output_root, "depth", _depth_visualization_filename(name)) for name in frame_names]
        write_mp4_from_png_sequence(depth_paths, os.path.join(output_root, "render_traj_depth.mp4"), fps)


def _depth_visualization_filename(output_name):
    stem, _ = os.path.splitext(output_name)
    return f"{stem}.png"


def _encode_depth_for_visualization(depth):
    depth = depth.detach().float()
    finite_mask = torch.isfinite(depth)
    if finite_mask.any():
        depth_min = depth[finite_mask].min()
        depth_max = depth[finite_mask].max()
    else:
        depth_min = torch.zeros((), dtype=depth.dtype, device=depth.device)
        depth_max = torch.zeros((), dtype=depth.dtype, device=depth.device)

    denom = depth_max - depth_min
    depth_norm = torch.zeros_like(depth)
    if float(denom.item()) > 1e-8:
        depth_norm[finite_mask] = ((depth[finite_mask] - depth_min) / (denom + 1e-8)).clamp(0.0, 1.0)
    if depth_norm.ndim == 2:
        depth_norm = depth_norm.unsqueeze(0)
    if depth_norm.shape[0] != 1:
        depth_norm = depth_norm[:1]
    depth_vis = torch.cat(
        [
            0.25 * depth_norm,
            0.65 * depth_norm,
            depth_norm,
        ],
        dim=0,
    )
    return depth_vis, float(depth_min.item()), float(depth_max.item())


def _save_depth_outputs(depth, depth_root, output_name):
    os.makedirs(depth_root, exist_ok=True)
    raw_depth_root = os.path.join(depth_root, "raw_depth")
    os.makedirs(raw_depth_root, exist_ok=True)

    depth_vis, depth_min, depth_max = _encode_depth_for_visualization(depth)
    vis_name = _depth_visualization_filename(output_name)
    raw_name = f"{output_name}.pt"
    vis_path = os.path.join(depth_root, vis_name)
    raw_path = os.path.join(raw_depth_root, raw_name)

    torchvision.utils.save_image(depth_vis, vis_path)
    torch.save(depth.detach().cpu(), raw_path)

    return {
        "view": output_name,
        "depth_visualization": vis_name,
        "raw_depth": os.path.join("raw_depth", raw_name),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "shape": [int(v) for v in depth.shape],
    }


def _write_depth_metadata(depth_root, entries):
    if not entries:
        return
    payload = {
        "normalization": {
            "formula": "d = (raw_depth - depth_min) / (depth_max - depth_min + eps)",
            "eps": 1e-8,
        },
        "visualization_encoding": {
            "type": "reversible_pseudocolor",
            "formula": "RGB = [0.25 * d, 0.65 * d, d]",
            "recover_normalized_depth": "d = blue_channel",
            "recover_raw_depth": "raw_depth = blue_channel * (depth_max - depth_min + eps) + depth_min",
        },
        "raw_depth_format": "torch.save tensor in .pt files",
        "views": entries,
    }
    with open(os.path.join(depth_root, "depth_metadata.json"), "w") as file:
        json.dump(payload, file, indent=2)


def compose_gt_with_background(view, background):
    gt = view.original_image[0:3, :, :].to(device=background.device, dtype=background.dtype)
    alpha_mask = getattr(view, "alpha_mask", None)
    if alpha_mask is None:
        return gt

    alpha = alpha_mask.to(device=gt.device, dtype=gt.dtype)
    bg = background.to(device=gt.device, dtype=gt.dtype).view(3, 1, 1)
    return gt * alpha + (1.0 - alpha) * bg


def _as_alpha_tensor(alpha):
    alpha = torch.clamp(alpha, 0.0, 1.0)
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    if alpha.ndim == 3 and alpha.shape[0] != 1:
        alpha = alpha[:1]
    return alpha


def _remove_background(color, alpha, background):
    alpha = _as_alpha_tensor(alpha).to(device=color.device, dtype=color.dtype)
    bg = background.to(device=color.device, dtype=color.dtype).view(3, 1, 1)
    unblended = (color - (1.0 - alpha) * bg) / alpha.clamp_min(1e-6)
    return torch.where(alpha > 1e-6, unblended, torch.zeros_like(unblended))


def _alpha_from_depth(depth, reference_image):
    alpha = (depth > 0).to(device=reference_image.device, dtype=reference_image.dtype)
    return _as_alpha_tensor(alpha)


def _save_image_maybe_transparent(image, path, args, alpha=None, background=None):
    if bool(getattr(args, "background_transparent", False)):
        if alpha is None:
            alpha = torch.ones((1, image.shape[-2], image.shape[-1]), dtype=image.dtype, device=image.device)
        alpha = _as_alpha_tensor(alpha).to(device=image.device, dtype=image.dtype)
        if alpha.shape[-2:] != image.shape[-2:]:
            raise RuntimeError(
                f"Alpha/image size mismatch for {path}: alpha {tuple(alpha.shape[-2:])}, "
                f"image {tuple(image.shape[-2:])}."
            )
        rgb = _remove_background(image, alpha, background) if background is not None else image
        rgba = torch.cat((torch.clamp(rgb, 0.0, 1.0), alpha), dim=0)
        torchvision.utils.save_image(rgba, path)
    else:
        torchvision.utils.save_image(image, path)


def render_rgb_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args):
    is_traj = name == "traj"
    output_root = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(output_root, "renders")
    gts_path = os.path.join(output_root, "gt")
    depth_path = os.path.join(output_root, "depth")
    os.makedirs(render_path, exist_ok=True)
    if not is_traj:
        os.makedirs(gts_path, exist_ok=True)

    output_names = _sequence_output_filenames(views) if is_traj else _view_output_filenames(views, name)
    depth_entries = []
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        output_name = output_names[idx]
        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            args,
            use_trained_exp=(train_test_exp and not is_traj),
            separate_sh=separate_sh,
            return_alpha=bool(getattr(args, "background_transparent", False)),
        )
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg.get("alpha")
        gt_alpha = getattr(view, "alpha_mask", None)

        if args.train_test_exp and not is_traj:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            depth = depth[..., depth.shape[-1] // 2:]
            if alpha is not None:
                alpha = alpha[..., alpha.shape[-1] // 2:]
            gt = compose_gt_with_background(view, background)
            gt = gt[..., gt.shape[-1] // 2:]
            if gt_alpha is not None:
                gt_alpha = gt_alpha.to(device=gt.device, dtype=gt.dtype)
                gt_alpha = gt_alpha[..., gt_alpha.shape[-1] // 2:]
        elif not is_traj:
            gt = compose_gt_with_background(view, background)
            if gt_alpha is not None:
                gt_alpha = gt_alpha.to(device=gt.device, dtype=gt.dtype)

        _save_image_maybe_transparent(rendering, os.path.join(render_path, output_name), args, alpha, background)
        if not is_traj:
            _save_image_maybe_transparent(gt, os.path.join(gts_path, output_name), args, gt_alpha, background)
        if bool(getattr(args, "render_depth", False)):
            depth_entries.append(_save_depth_outputs(depth, depth_path, output_name))

    if bool(getattr(args, "render_depth", False)):
        _write_depth_metadata(depth_path, depth_entries)
    if is_traj:
        write_traj_videos(
            output_root,
            output_names,
            fps=int(getattr(args, "render_path_fps", 60)),
            include_depth=bool(getattr(args, "render_depth", False)),
        )


def render_single_object_masks(model_path, name, views, gaussians, pipeline, background, train_test_exp, args):
    is_traj = name == "traj"
    render_path = os.path.join(model_path, name, f"ours_{str(args.N4views) + 'x'}", "mask_renders", args.text)
    image_path = os.path.join(model_path, name, f"ours_{str(args.N4views) + 'x'}", "image_renders", args.text)
    depth_path = os.path.join(model_path, name, f"ours_{str(args.N4views) + 'x'}", "depth_renders", args.text)
    depth_export_path = os.path.join(model_path, name, f"ours_{str(args.N4views) + 'x'}", "depth", args.text)

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    output_names = _sequence_output_filenames(views) if is_traj else _view_output_filenames(views, name)
    depth_entries = []
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        output_name = output_names[idx]
        renders = render(view, gaussians, pipeline, background, args, use_trained_exp=(train_test_exp and not is_traj))
        depth = renders["depth"]
        mask = renders["mask"]
        render_image = renders["render"]
        alpha = _alpha_from_depth(depth, render_image)
        mask = (mask > 0.5).float()[0, :, :]
        depth_vis, _, _ = _encode_depth_for_visualization(depth)
        torchvision.utils.save_image(depth_vis, os.path.join(depth_path, output_name))
        _save_image_maybe_transparent(render_image, os.path.join(image_path, output_name), args, alpha, background)
        torchvision.utils.save_image(mask, os.path.join(render_path, output_name))
        if bool(getattr(args, "render_depth", False)):
            depth_entries.append(_save_depth_outputs(depth, depth_export_path, output_name))

    if bool(getattr(args, "render_depth", False)):
        _write_depth_metadata(depth_export_path, depth_entries)
    if is_traj:
        write_mp4_from_png_sequence(
            [os.path.join(image_path, name) for name in output_names],
            os.path.join(os.path.dirname(image_path), "render_traj_color.mp4"),
            fps=int(getattr(args, "render_path_fps", 60)),
        )
        write_mp4_from_png_sequence(
            [os.path.join(depth_path, name) for name in output_names],
            os.path.join(os.path.dirname(depth_path), "render_traj_depth.mp4"),
            fps=int(getattr(args, "render_path_fps", 60)),
        )


def render_multi_object_outputs(model_path, split_name, views, gaussians, pipeline, background, train_test_exp, args, labels, desk_object_id, id_in_background):
    render_dir_name = "decouple" if bool(getattr(args, "only_removal", False)) else "decouple+inpaint"
    render_root = os.path.join(model_path, render_dir_name)
    background_filter = gaussians.get_object_filter(0)
    is_traj = split_name == "traj"
    output_names = _sequence_output_filenames(views) if is_traj else _view_output_filenames(views, split_name)
    render_targets = []
    if not bool(getattr(args, "only_desk_background", False)):
        render_targets.append(("original", None))
        render_targets.extend(
            (f"object_{label}", gaussians.get_object_filter(label))
            for label in labels
            if int(label) != int(desk_object_id)
        )
    desk_background_filter = torch.logical_or(gaussians.get_object_filter(desk_object_id), background_filter)
    for object_id in id_in_background:
        desk_background_filter = torch.logical_or(desk_background_filter, gaussians.get_object_filter(object_id))
    desk_background_name = "desk+background"
    if id_in_background:
        desk_background_name += "_" + "_".join(str(int(object_id)) for object_id in id_in_background)
    render_targets.append((desk_background_name, desk_background_filter))

    for target_name, gaussian_filter in render_targets:
        object_root = os.path.join(render_root, target_name, split_name)
        image_root = os.path.join(object_root, "render")
        depth_root = os.path.join(object_root, "depth")
        os.makedirs(image_root, exist_ok=True)
        depth_entries = []
        selected_count = int(gaussians.get_xyz.shape[0]) if gaussian_filter is None else int(gaussian_filter.sum().item())
        for idx, view in enumerate(tqdm(views, desc=f"Rendering {target_name} {split_name}")):
            output_name = output_names[idx]
            if selected_count == 0:
                render_image = background[:, None, None].expand(3, int(view.image_height), int(view.image_width))
                depth = torch.zeros((1, int(view.image_height), int(view.image_width)), dtype=torch.float32, device=background.device)
                alpha = torch.zeros((1, int(view.image_height), int(view.image_width)), dtype=render_image.dtype, device=render_image.device)
            else:
                mask_override = torch.ones((selected_count,), device=background.device)
                renders = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    args,
                    use_trained_exp=(train_test_exp and not is_traj),
                    gaussian_filter=gaussian_filter,
                    mask_override=mask_override,
                    return_alpha=bool(getattr(args, "background_transparent", False)),
                )
                render_image = renders["render"]
                depth = renders["depth"]
                alpha = renders.get("alpha")
                if alpha is None:
                    alpha = _alpha_from_depth(depth, render_image)
            _save_image_maybe_transparent(render_image, os.path.join(image_root, output_name), args, alpha, background)
            if bool(getattr(args, "render_depth", False)):
                depth_entries.append(_save_depth_outputs(depth, depth_root, output_name))

        if bool(getattr(args, "render_depth", False)):
            _write_depth_metadata(depth_root, depth_entries)
        if is_traj:
            write_traj_videos(
                object_root,
                output_names,
                fps=int(getattr(args, "render_path_fps", 60)),
                include_depth=bool(getattr(args, "render_depth", False)),
                render_subdir="render",
            )


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, separate_sh):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
        render_path = bool(getattr(args, "render_path", False))

        if not args.include_mask:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            if not skip_train:
                render_rgb_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)
            if not skip_test:
                render_rgb_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)
            if render_path:
                traj_views = generate_render_path(scene_render_path_cameras(scene), int(getattr(args, "render_path_frames", 240)))
                render_rgb_set(dataset.model_path, "traj", scene.loaded_iter, traj_views, gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)
            return

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        args.N4views, _ = resolve_segmentation_n4views(dataset, getattr(args, "N4views", None))

        if dataset.mask_mode == "multi_label":
            checkpoint, model_params, checkpoint_marker = _resolve_multi_object_checkpoint(dataset.model_path, scene.loaded_iter, args)
            _restore_or_apply_multi_object_state(gaussians, model_params, checkpoint, args)
            print(f"Loaded multi-object state from {checkpoint}")
            if checkpoint_marker is not None:
                print(f"Object label checkpoint marker: {checkpoint_marker}")
            assigned_labels = sorted(set(gaussians.get_assigned_object_labels()))
            desk_object_id = _infer_desk_object_id(assigned_labels, getattr(args, "desk_object_id", None))
            print(f"Using desk_object_id={desk_object_id}")
            metadata_path = os.path.join(dataset.model_path, "multi_object", "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as file:
                    metadata = json.load(file)
                labels = [int(label) for label in metadata.get("ordered_labels", []) if int(label) in assigned_labels]
                labels.extend(label for label in assigned_labels if label not in labels)
            else:
                labels = list(assigned_labels)
            id_in_background = _resolve_id_in_background(args, assigned_labels)
            if id_in_background:
                print(f"Adding ids to desk+background: {id_in_background}")
            if bool(getattr(args, "only_desk_background", False)):
                print("Rendering only desk+background targets.")
            if not skip_train:
                render_multi_object_outputs(dataset.model_path, "train", scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, args, labels, desk_object_id, id_in_background)
            if not skip_test:
                render_multi_object_outputs(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, args, labels, desk_object_id, id_in_background)
            if render_path:
                traj_views = generate_render_path(scene_render_path_cameras(scene), int(getattr(args, "render_path_frames", 240)))
                render_multi_object_outputs(dataset.model_path, "traj", traj_views, gaussians, pipeline, background, dataset.train_test_exp, args, labels, desk_object_id, id_in_background)
            return

        if not dataset.mask_path:
            raise ValueError("Single-object rendering requires --text so the segmentation checkpoint directory can be resolved.")
        checkpoint = os.path.join(dataset.mask_path, f"chkpnt{len(scene.getTrainCameras()) * args.N4views}.pth")
        model_params, _ = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode="test")
        gaussians.segment()
        if not skip_train:
            render_single_object_masks(dataset.model_path, "train", scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, args)
        if not skip_test:
            render_single_object_masks(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, args)
        if render_path:
            traj_views = generate_render_path(scene_render_path_cameras(scene), int(getattr(args, "render_path_frames", 240)))
            render_single_object_masks(dataset.model_path, "traj", traj_views, gaussians, pipeline, background, dataset.train_test_exp, args)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--N4views", type=int, default=None)
    parser.add_argument("--include_mask", action="store_true")
    parser.add_argument("--finetune_mask", action="store_true")
    parser.add_argument("--desk_object_id", default=None, type=int)
    parser.add_argument("--segmentation_checkpoint", default=None, type=str)
    parser.add_argument("--id_in_background", nargs="+", default=None)
    parser.add_argument("--only_desk+background", "--only_desk_background", dest="only_desk_background", action="store_true", default=False)
    parser.add_argument("--only_removal", action="store_true", default=False)
    parser.add_argument("--render_depth", action="store_true", default=False)
    parser.add_argument("--background_transparent", action="store_true", default=False)
    parser.add_argument("--render_path", action="store_true", default=False)
    parser.add_argument("--render_path_frames", default=240, type=int)
    parser.add_argument("--render_path_fps", default=60, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
