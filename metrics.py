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

from pathlib import Path
import os
import shutil
import tempfile
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _load_rgb_tensor(path):
    with Image.open(path) as image:
        return tf.to_tensor(image.convert("RGB")).unsqueeze(0)[:, :3, :, :].cuda()


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        gt_path = gt_dir / fname
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground-truth image not found for render {fname}: {gt_path}")
        renders.append(_load_rgb_tensor(renders_dir / fname))
        gts.append(_load_rgb_tensor(gt_path))
        image_names.append(fname)
    return renders, gts, image_names


def _read_split_list(source_path, split_name):
    split_path = Path(source_path) / f"{split_name}_list.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_name}_list.txt not found: {split_path}")
    names = []
    for line in split_path.read_text().splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        names.append(os.path.basename(name))
    if not names:
        raise RuntimeError(f"No image names were found in {split_path}")
    return names


def _resolve_image_path(root_dir, image_name, desc):
    root_dir = Path(root_dir)
    basename = os.path.basename(str(image_name).strip())
    exact_path = root_dir / basename
    if exact_path.exists():
        return exact_path

    stem = Path(basename).stem
    matches = sorted(
        path for path in root_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == stem
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous {desc} image for '{image_name}' under {root_dir}: {matches}")
    raise FileNotFoundError(f"{desc} image not found for '{image_name}' under {root_dir}")


def readAfterInpaintImages(model_path, source_path):
    renders_dir = Path(model_path) / "decouple" / "desk+background" / "test" / "render"
    gt_dir = Path(source_path) / "removal_GT"
    if not renders_dir.exists():
        raise FileNotFoundError(f"After-inpaint render directory not found: {renders_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"After-inpaint GT directory not found: {gt_dir}")

    renders = []
    gts = []
    image_names = []
    render_paths = []
    gt_paths = []
    for image_name in _read_split_list(source_path, "test"):
        render_path = _resolve_image_path(renders_dir, image_name, "after-inpaint render")
        gt_path = _resolve_image_path(gt_dir, image_name, "removal_GT")
        render = _load_rgb_tensor(render_path)
        gt = _load_rgb_tensor(gt_path)
        if tuple(render.shape) != tuple(gt.shape):
            raise RuntimeError(
                f"Image shape mismatch for '{image_name}': render={tuple(render.shape)} "
                f"gt={tuple(gt.shape)}"
            )
        renders.append(render)
        gts.append(gt)
        image_names.append(render_path.name)
        render_paths.append(render_path)
        gt_paths.append(gt_path)

    return renders, gts, image_names, render_paths, gt_paths


def _stage_images_for_fid(image_paths, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, source_path in enumerate(image_paths):
        suffix = source_path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".png"
        target_path = output_dir / f"{idx:05d}{suffix}"
        try:
            os.symlink(source_path.resolve(), target_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def calculate_fid_for_pairs(gt_paths, render_paths, batch_size=8):
    with tempfile.TemporaryDirectory(prefix="mod_cob_gs_after_inpaint_fid_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        gt_subset = tmp_root / "gt"
        render_subset = tmp_root / "renders"
        _stage_images_for_fid(gt_paths, gt_subset)
        _stage_images_for_fid(render_paths, render_subset)
        return calculate_fid(gt_subset, render_subset, batch_size=batch_size)


def calculate_fid(gt_dir, renders_dir, batch_size=8):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise ImportError(
            "FID computation requires the PyPI package 'pytorch-fid' "
            "(import name: pytorch_fid). Install it with: python -m pip install pytorch-fid"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return calculate_fid_given_paths(
        [str(gt_dir), str(renders_dir)],
        batch_size,
        device,
        2048,
        8,
    )


def write_qualitative_comparison(model_path, scene_results, output_name="qualitative_comparison.txt"):
    metric_order = ("PSNR", "SSIM", "LPIPS", "FID")
    present_metrics = [
        metric for metric in metric_order
        if any(metric in method_results for method_results in scene_results.values())
    ]

    output_path = Path(model_path) / output_name
    lines = ["method\t" + "\t".join(present_metrics)]
    for method in sorted(scene_results):
        row = [method]
        for metric in present_metrics:
            value = scene_results[method].get(metric)
            row.append("N/A" if value is None else f"{float(value):.7f}")
        lines.append("\t".join(row))

    with open(output_path, "w") as file:
        file.write("\n".join(lines) + "\n")


def evaluate(model_paths, method_names=None, compute_fid=True):

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"
        selected_methods = set(method_names) if method_names is not None else None
        found_method = False

        for method in sorted(os.listdir(test_dir)):
            if selected_methods is not None and method not in selected_methods:
                continue
            found_method = True
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)
            if not renders:
                raise RuntimeError(f"No rendered images found under {renders_dir}")

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

            ssim_values = torch.tensor(ssims).tolist()
            psnr_values = torch.tensor(psnrs).tolist()
            lpips_values = torch.tensor(lpipss).tolist()
            mean_ssim = torch.tensor(ssims).mean().item()
            mean_psnr = torch.tensor(psnrs).mean().item()
            mean_lpips = torch.tensor(lpipss).mean().item()
            method_results = {
                "SSIM": mean_ssim,
                "PSNR": mean_psnr,
                "LPIPS": mean_lpips,
            }

            per_view_dict[scene_dir][method].update({
                "SSIM": {name: value for value, name in zip(ssim_values, image_names)},
                "PSNR": {name: value for value, name in zip(psnr_values, image_names)},
                "LPIPS": {name: value for value, name in zip(lpips_values, image_names)},
            })

            del renders, gts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if compute_fid:
                method_results["FID"] = float(calculate_fid(gt_dir, renders_dir))

            print("  SSIM : {:>12.7f}".format(mean_ssim))
            print("  PSNR : {:>12.7f}".format(mean_psnr))
            print("  LPIPS: {:>12.7f}".format(mean_lpips))
            if compute_fid:
                print("  FID  : {:>12.7f}".format(method_results["FID"]))
            print("")

            full_dict[scene_dir][method].update(method_results)

        if selected_methods is not None and not found_method:
            raise FileNotFoundError(f"None of the requested methods were found under {test_dir}: {sorted(selected_methods)}")

        with open(scene_dir + "/results.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        write_qualitative_comparison(scene_dir, full_dict[scene_dir])


def _evaluate_image_lists(renders, gts, image_names):
    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

    ssim_values = torch.tensor(ssims).tolist()
    psnr_values = torch.tensor(psnrs).tolist()
    lpips_values = torch.tensor(lpipss).tolist()
    method_results = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
    }
    per_view_results = {
        "SSIM": {name: value for value, name in zip(ssim_values, image_names)},
        "PSNR": {name: value for value, name in zip(psnr_values, image_names)},
        "LPIPS": {name: value for value, name in zip(lpips_values, image_names)},
    }
    return method_results, per_view_results


def evaluate_after_inpaint(model_paths, source_paths, compute_fid=True):
    if len(source_paths) == 1 and len(model_paths) > 1:
        source_paths = source_paths * len(model_paths)
    if len(source_paths) != len(model_paths):
        raise ValueError(
            "--after_inpaint requires either one --source_path for all model paths or "
            "the same number of source paths as model paths."
        )

    full_dict = {}
    per_view_dict = {}
    method = "desk+background"
    print("")

    for scene_dir, source_path in zip(model_paths, source_paths):
        print("Scene:", scene_dir)
        print("Source:", source_path)
        print("Method:", method)

        renders, gts, image_names, render_paths, gt_paths = readAfterInpaintImages(scene_dir, source_path)
        if not renders:
            raise RuntimeError(f"No after-inpaint rendered images found for {scene_dir}")

        method_results, per_view_results = _evaluate_image_lists(renders, gts, image_names)
        del renders, gts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if compute_fid:
            method_results["FID"] = float(calculate_fid_for_pairs(gt_paths, render_paths))

        print("  SSIM : {:>12.7f}".format(method_results["SSIM"]))
        print("  PSNR : {:>12.7f}".format(method_results["PSNR"]))
        print("  LPIPS: {:>12.7f}".format(method_results["LPIPS"]))
        if compute_fid:
            print("  FID  : {:>12.7f}".format(method_results["FID"]))
        print("")

        full_dict[scene_dir] = {method: method_results}
        per_view_dict[scene_dir] = {method: per_view_results}

        with open(Path(scene_dir) / "results_after_inpaint.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(Path(scene_dir) / "per_view_after_inpaint.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
        write_qualitative_comparison(
            scene_dir,
            full_dict[scene_dir],
            output_name="qualitative_comparison_after_inpaint.txt",
        )

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--source_path', '-s', nargs="+", type=str, default=None)
    parser.add_argument('--after_inpaint', action="store_true", default=False)
    args = parser.parse_args()
    if args.after_inpaint:
        if args.source_path is None:
            parser.error("--after_inpaint requires --source_path / -s")
        evaluate_after_inpaint(args.model_paths, args.source_path)
    else:
        evaluate(args.model_paths)
