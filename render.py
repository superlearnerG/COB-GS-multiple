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
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.mask_provider import parse_target_labels

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


def render_rgb_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, args):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        rendering = render(view, gaussians, pipeline, background, args, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))


def render_single_object_masks(model_path, name, views, gaussians, pipeline, background, train_test_exp, args):
    render_path = os.path.join(model_path, name, f"ours_{str(args.N4views) + 'x'}", "mask_renders", args.text)
    image_path = os.path.join(model_path, name, f"ours_{str(args.N4views) + 'x'}", "image_renders", args.text)
    depth_path = os.path.join(model_path, name, f"ours_{str(args.N4views) + 'x'}", "depth_renders", args.text)

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        renders = render(view, gaussians, pipeline, background, args, use_trained_exp=train_test_exp)
        depth = renders["depth"]
        mask = renders["mask"]
        render_image = renders["render"]
        mask = (mask > 0.5).float()[0, :, :]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        torchvision.utils.save_image(depth, os.path.join(depth_path, f"depth_{idx:05d}.png"))
        torchvision.utils.save_image(render_image, os.path.join(image_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(mask, os.path.join(render_path, f"{idx:05d}.png"))


def render_multi_object_outputs(model_path, split_name, views, gaussians, pipeline, background, train_test_exp, args, labels):
    render_root = os.path.join(model_path, "decouple")
    background_filter = gaussians.get_object_filter(0)
    render_targets = [(f"object_{label}", gaussians.get_object_filter(label)) for label in labels]
    render_targets.append(("background", background_filter))

    desk_label = None
    if labels:
        desk_label = 255 if 255 in labels else max(labels)
        desk_background_filter = torch.logical_or(gaussians.get_object_filter(desk_label), background_filter)
        render_targets.append(("desk+background", desk_background_filter))

    for target_name, gaussian_filter in render_targets:
        object_root = os.path.join(render_root, target_name, split_name)
        image_root = os.path.join(object_root, "render")
        os.makedirs(image_root, exist_ok=True)
        selected_count = int(gaussian_filter.sum().item())
        for idx, view in enumerate(tqdm(views, desc=f"Rendering {target_name} {split_name}")):
            if selected_count == 0:
                render_image = background[:, None, None].expand(3, int(view.image_height), int(view.image_width))
            else:
                mask_override = torch.ones((selected_count,), device="cuda")
                renders = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    args,
                    use_trained_exp=train_test_exp,
                    gaussian_filter=gaussian_filter,
                    mask_override=mask_override,
                )
                render_image = renders["render"]
            torchvision.utils.save_image(render_image, os.path.join(image_root, f"{idx:05d}.png"))


def render_sets(dataset, iteration, pipeline, skip_train, skip_test, separate_sh):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(background_color, dtype=torch.float32, device="cuda")

        if not args.include_mask:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            if not skip_train:
                render_rgb_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)
            if not skip_test:
                render_rgb_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, args)
            return

        scene = Scene(dataset, gaussians, shuffle=False)
        args.N4views, _ = resolve_segmentation_n4views(dataset, args.N4views)

        if dataset.mask_mode == "multi_label":
            checkpoint = os.path.join(dataset.model_path, "multi_object", "final_multi_object.pth")
            model_params, _ = torch.load(checkpoint)
            gaussians.restore(model_params, args, mode="test")
            requested_labels = parse_target_labels(dataset.target_labels)
            assigned_labels = set(gaussians.get_assigned_object_labels())
            metadata_path = os.path.join(dataset.model_path, "multi_object", "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as file:
                    metadata = json.load(file)
                labels = [int(label) for label in metadata.get("ordered_labels", []) if int(label) in assigned_labels]
            else:
                labels = sorted(assigned_labels)
            if requested_labels:
                labels = [label for label in labels if label in requested_labels]
            if not skip_train:
                render_multi_object_outputs(dataset.model_path, "train", scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, args, labels)
            if not skip_test:
                render_multi_object_outputs(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, args, labels)
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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
