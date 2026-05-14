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
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.mask_provider import MultiLabelMaskProvider, SingleTextMaskProvider, parse_target_labels
from utils.multi_object_postprocess import (
    postprocess_committed_object,
    render_postprocess_debug,
    resolve_debug_views,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

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


def resolve_segmentation_checkpoint(model_path, checkpoint):
    if checkpoint is not None:
        return checkpoint
    default_checkpoint = os.path.join(model_path, "chkpnt30000.pth")
    if os.path.exists(default_checkpoint):
        return default_checkpoint
    raise FileNotFoundError("Segmentation mode requires a base RGB checkpoint. Expected --start_checkpoint or model_path/chkpnt30000.pth.")


def save_checkpoint(path, gaussians, marker, include_mask):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save((gaussians.capture(include_mask), marker), path)


def write_multi_object_metadata(path, metadata):
    with open(path, "w") as file:
        json.dump(metadata, file, indent=2)


def build_mask_provider(dataset, scene):
    if dataset.mask_mode == "multi_label":
        return MultiLabelMaskProvider(
            dataset.mask_root,
            scene.getTrainCameras(),
            target_labels=dataset.target_labels,
            object_order=dataset.object_order,
        )
    if not dataset.mask_path:
        raise ValueError("Single-object segmentation requires --text so that the mask directory can be resolved.")
    return SingleTextMaskProvider(dataset.mask_path)


def compute_rgb_loss(image, gt_image, render_pkg, viewpoint_cam, depth_l1_weight_value, opt, use_depth_loss):
    ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value)

    ll1depth = 0.0
    if use_depth_loss and depth_l1_weight_value > 0 and viewpoint_cam.depth_reliable:
        inv_depth = render_pkg["depth"]
        mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        depth_mask = viewpoint_cam.depth_mask.cuda()
        valid_pixels = depth_mask.sum().clamp_min(1.0)
        ll1depth_pure = (torch.abs(inv_depth - mono_invdepth) * depth_mask).sum() / valid_pixels
        ll1depth = depth_l1_weight_value * ll1depth_pure
        loss += ll1depth
        ll1depth = ll1depth.item()
    return ll1, loss, ll1depth


def compose_gt_with_background(viewpoint_cam, background):
    gt_image = viewpoint_cam.original_image.cuda()
    alpha_mask = getattr(viewpoint_cam, "alpha_mask", None)
    if alpha_mask is None:
        return gt_image

    alpha = alpha_mask.to(device=gt_image.device, dtype=gt_image.dtype)
    bg = background.to(device=gt_image.device, dtype=gt_image.dtype).view(3, 1, 1)
    return gt_image * alpha + (1.0 - alpha) * bg


def maybe_handle_viewer(iteration, dataset, pipe, gaussians, background, opt):
    if network_gui.conn is None:
        network_gui.try_connect()
    while network_gui.conn is not None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam is not None:
                net_image = render(
                    custom_cam,
                    gaussians,
                    pipe,
                    background,
                    opt,
                    scaling_modifier=scaling_modifer,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, dataset.source_path)
            if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                break
        except Exception:
            network_gui.conn = None


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


def render_and_evaluate_vanilla_test_set(dataset, opt, pipe, scene, gaussians, background):
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("\n[Vanilla Eval] No test cameras found; skipping PSNR/SSIM/LPIPS/FID evaluation.")
        return

    method_name = f"ours_{opt.iterations}"
    render_path = os.path.join(scene.model_path, "test", method_name, "renders")
    gt_path = os.path.join(scene.model_path, "test", method_name, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    print(f"\n[Vanilla Eval] Rendering {len(test_cameras)} test views to {os.path.join(scene.model_path, 'test', method_name)}")
    output_names = _view_output_filenames(test_cameras, "vanilla test")
    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Rendering vanilla test set")):
            output_name = output_names[idx]
            rendering = render(
                viewpoint,
                gaussians,
                pipe,
                background,
                opt,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"]
            gt_image = compose_gt_with_background(viewpoint, background)

            if dataset.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt_image = gt_image[..., gt_image.shape[-1] // 2:]

            torchvision.utils.save_image(torch.clamp(rendering, 0.0, 1.0), os.path.join(render_path, output_name))
            torchvision.utils.save_image(torch.clamp(gt_image, 0.0, 1.0), os.path.join(gt_path, output_name))

    torch.cuda.empty_cache()
    from metrics import evaluate as evaluate_metrics

    evaluate_metrics([scene.model_path], method_names=[method_name], compute_fid=True)


def ensure_vanilla_eval_dependencies(scene):
    if not scene.getTestCameras():
        return
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Vanilla test evaluation requires the PyPI package 'pytorch-fid' "
            "(import name: pytorch_fid). Install it with: python -m pip install pytorch-fid"
        ) from exc


def run_rgb_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    ensure_vanilla_eval_dependencies(scene)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        maybe_handle_viewer(iteration, dataset, pipe, gaussians, background, opt)
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, opt, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = compose_gt_with_background(viewpoint_cam, bg)
        ll1, loss, ll1depth = compute_rgb_loss(
            image,
            gt_image,
            render_pkg,
            viewpoint_cam,
            depth_l1_weight(iteration),
            opt,
            getattr(dataset, "use_depth_loss", False),
        )
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_ll1depth_for_log = 0.4 * ll1depth + 0.6 * ema_ll1depth_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Depth Loss": f"{ema_ll1depth_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                0,
                ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, opt, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                save_checkpoint(os.path.join(scene.model_path, f"chkpnt{iteration}.pth"), gaussians, iteration, False)

    render_and_evaluate_vanilla_test_set(dataset, opt, pipe, scene, gaussians, background)


def run_single_object_stage(dataset, opt, pipe, scene, gaussians, mask_provider, target_label=None, progress_desc="Segmentation progress", tb_writer=None, log_iteration_offset=0, debug_from=-1):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    base_num = len(viewpoint_indices) * 2
    ema_loss_for_log = 0.0
    ema_ll1depth_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc=progress_desc)
    mask_state = False if opt.finetune_mask else True

    for iteration in range(1, opt.iterations + 1):
        maybe_handle_viewer(iteration, dataset, pipe, gaussians, background, opt)
        iter_start.record()

        if opt.finetune_mask and iteration % base_num == 1:
            mask_state = not mask_state
            gaussians.add_training_state(mask_state)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, opt, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]

        if mask_state:
            rendered_mask = render_pkg["mask"]
            mask_signals = render_pkg["mask_signals"]
            gt_mask = mask_provider.get_mask(viewpoint_cam, target_label)
            loss_mask = (-(gt_mask * rendered_mask).sum() + opt.lamb * ((1 - gt_mask) * rendered_mask).sum()) / (gt_mask.sum() + (1 - gt_mask).sum())
            ll1 = loss_mask
            loss = loss_mask
            ll1depth = 0
        else:
            gt_image = compose_gt_with_background(viewpoint_cam, bg)
            ll1, loss, ll1depth = compute_rgb_loss(
                image,
                gt_image,
                render_pkg,
                viewpoint_cam,
                depth_l1_weight(iteration),
                opt,
                getattr(dataset, "use_depth_loss", False),
            )

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_ll1depth_for_log = 0.4 * ll1depth + 0.6 * ema_ll1depth_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "Depth Loss": f"{ema_ll1depth_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar("segmentation/loss", loss.item(), log_iteration_offset + iteration)
                if mask_state:
                    tb_writer.add_scalar("segmentation/mask_loss", loss.item(), log_iteration_offset + iteration)
                else:
                    tb_writer.add_scalar("segmentation/rgb_loss", loss.item(), log_iteration_offset + iteration)

            if mask_state:
                gaussians.add_mask_signal_densification_stats(mask_signals)
                if opt.finetune_mask and iteration % base_num == 0:
                    print("mask_sig_split")
                    if iteration < opt.iterations:
                        gaussians.mask_and_split(opt.mask_signals_threshold, scene.cameras_extent, base_num)
                    else:
                        gaussians.mask_and_split(opt.mask_signals_threshold, scene.cameras_extent, base_num, prune_only=True)

            if iteration < opt.iterations:
                if mask_state:
                    gaussians.mask_optimizer.step()
                    gaussians.mask_optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    if SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
                        radii = render_pkg["radii"]
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)


def run_single_text_segmentation(dataset, opt, pipe, n4views, checkpoint, debug_from):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    opt.iterations = len(scene.getTrainCameras()) * n4views

    checkpoint = resolve_segmentation_checkpoint(dataset.model_path, checkpoint)
    model_params, _ = torch.load(checkpoint)
    gaussians.restore(model_params, opt)

    mask_provider = build_mask_provider(dataset, scene)
    run_single_object_stage(
        dataset,
        opt,
        pipe,
        scene,
        gaussians,
        mask_provider,
        target_label=None,
        progress_desc="Single-object segmentation",
        tb_writer=tb_writer,
        debug_from=debug_from,
    )

    save_checkpoint(os.path.join(dataset.mask_path, f"chkpnt{opt.iterations}.pth"), gaussians, opt.iterations, True)


def run_multi_object_segmentation(dataset, opt, pipe, n4views, checkpoint, scene_type, debug_from):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    opt.iterations = len(scene.getTrainCameras()) * n4views
    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")

    checkpoint = resolve_segmentation_checkpoint(dataset.model_path, checkpoint)
    model_params, _ = torch.load(checkpoint)
    gaussians.restore(model_params, opt)

    mask_provider = build_mask_provider(dataset, scene)
    labels = mask_provider.get_object_labels()
    area_by_label = mask_provider.get_area_by_label()

    multi_object_root = os.path.join(dataset.model_path, "multi_object")
    os.makedirs(multi_object_root, exist_ok=True)
    metadata_path = os.path.join(multi_object_root, "metadata.json")

    if opt.object_debug_views and not opt.enable_object_postprocess:
        raise ValueError("Debug postprocess renders require --enable_object_postprocess.")
    debug_views = resolve_debug_views(scene, opt.object_debug_views) if opt.enable_object_postprocess else []
    skip_postprocess_labels = set(parse_target_labels(getattr(opt, "object_postprocess_skip_labels", "")))

    metadata = {
        "scene_type": scene_type,
        "n4views": n4views,
        "ordered_labels": labels,
        "object_order": dataset.object_order,
        "mask_root": dataset.mask_root,
        "target_labels": labels,
        "area_by_label": {str(label): int(area_by_label.get(label, 0)) for label in labels},
        "commit_threshold": 0.90,
        "committed_count": {},
        "postprocess": {
            "enabled": bool(opt.enable_object_postprocess),
            "skip_labels": sorted(skip_postprocess_labels),
            "debug_views": [view.image_name for view in debug_views],
            "labels": {},
        },
    }

    write_multi_object_metadata(metadata_path, metadata)

    log_iteration_offset = 0
    for label in labels:
        print(f"\n[OBJECT {label}] Resetting segmentation state")
        gaussians.training_setup(opt)
        gaussians.reset_mask()

        run_single_object_stage(
            dataset,
            opt,
            pipe,
            scene,
            gaussians,
            mask_provider,
            target_label=label,
            progress_desc=f"Object {label} segmentation",
            tb_writer=tb_writer,
            log_iteration_offset=log_iteration_offset,
            debug_from=debug_from,
        )
        committed = gaussians.commit_current_object(label, commit_thresh=0.90)
        metadata["committed_count"][str(label)] = committed
        print(f"[OBJECT {label}] Committed {committed} Gaussians")

        if opt.enable_object_postprocess:
            if int(label) in skip_postprocess_labels:
                object_count = int((gaussians.get_object_id == int(label)).sum().item())
                postprocess_stats = {
                    "object_count_before": object_count,
                    "object_count_after": object_count,
                    "floaters_unassigned": 0,
                    "background_pruned": 0,
                    "voxel_size": None,
                    "cleanup_skipped_reason": "label_in_object_postprocess_skip_labels",
                }
                metadata["postprocess"]["labels"][str(label)] = postprocess_stats
                print(f"[OBJECT {label}] Skipping postprocess because label is in --object_postprocess_skip_labels")
                write_multi_object_metadata(metadata_path, metadata)
                log_iteration_offset += opt.iterations
                continue

            mask_response = gaussians.get_mask.detach().squeeze().clone()
            postprocess_stats = postprocess_committed_object(
                gaussians,
                label,
                scene.cameras_extent,
                mask_response,
                opt,
            )
            metadata["postprocess"]["labels"][str(label)] = postprocess_stats
            print(
                f"[OBJECT {label}] Postprocess kept {postprocess_stats['object_count_after']} points, "
                f"unassigned {postprocess_stats['floaters_unassigned']}, "
                f"pruned {postprocess_stats['background_pruned']}"
            )
            if debug_views:
                render_postprocess_debug(
                    dataset.model_path,
                    label,
                    debug_views,
                    gaussians,
                    pipe,
                    background,
                    dataset.train_test_exp,
                    opt,
                    SPARSE_ADAM_AVAILABLE,
                )

        write_multi_object_metadata(metadata_path, metadata)
        log_iteration_offset += opt.iterations

    final_checkpoint = os.path.join(multi_object_root, "final_multi_object.pth")
    save_checkpoint(final_checkpoint, gaussians, len(labels), True)
    write_multi_object_metadata(metadata_path, metadata)


def training(dataset, opt, pipe, n4views, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    if not opt.include_mask:
        run_rgb_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)
        return

    n4views, scene_type = resolve_segmentation_n4views(dataset, n4views)
    if dataset.mask_mode == "multi_label":
        run_multi_object_segmentation(dataset, opt, pipe, n4views, checkpoint, scene_type, debug_from)
    else:
        run_single_text_segmentation(dataset, opt, pipe, n4views, checkpoint, debug_from)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, base_num, ll1, loss, l1_loss_fn, elapsed, testing_iterations, scene, render_func, render_args, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(render_func(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(compose_gt_with_background(viewpoint, render_args[1]), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and idx < 5:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--N4views", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.N4views,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    print("\nTraining complete.")
