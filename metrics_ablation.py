import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ablation image metrics by matching images in --input_path "
            "to same-name ground-truth images in --GT_path."
        )
    )
    parser.add_argument(
        "--input_path",
        "--input-path",
        "-i",
        required=True,
        help="Directory containing images to evaluate. This may be a subset of --GT_path.",
    )
    parser.add_argument(
        "--GT_path",
        "--gt_path",
        "--gt-path",
        "-g",
        dest="gt_path",
        required=True,
        help="Directory containing ground-truth images with matching filenames.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for metric computation. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--skip_lpips",
        "--skip-lpips",
        action="store_true",
        help="Skip LPIPS if only PSNR/SSIM are needed.",
    )
    parser.add_argument(
        "--per_image",
        "--per-image",
        action="store_true",
        help="Print per-image metrics in addition to means.",
    )
    parser.add_argument(
        "--compute_fid",
        "--compute-fid",
        "--fid",
        action="store_true",
        help="Compute FID on the matched input/GT subset.",
    )
    parser.add_argument(
        "--fid_batch_size",
        "--fid-batch-size",
        type=int,
        default=8,
        help="Batch size for pytorch-fid. Defaults to 8.",
    )
    return parser.parse_args()


def require_dir(path: str, desc: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{desc} directory not found: {resolved}")
    return resolved


def list_input_images(input_path: Path) -> List[Path]:
    image_paths = [
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths, key=lambda path: path.name)


def build_pairs(input_path: Path, gt_path: Path) -> List[Tuple[Path, Path, str]]:
    input_images = list_input_images(input_path)
    if not input_images:
        raise FileNotFoundError(f"No supported images found under --input_path: {input_path}")

    pairs: List[Tuple[Path, Path, str]] = []
    missing: List[str] = []
    for input_image in input_images:
        gt_image = gt_path / input_image.name
        if gt_image.is_file():
            pairs.append((input_image, gt_image, input_image.name))
        else:
            missing.append(input_image.name)

    if missing:
        missing_preview = ", ".join(missing[:20])
        if len(missing) > 20:
            missing_preview += f", ... ({len(missing)} total)"
        raise FileNotFoundError(
            "Ground-truth images missing for input filenames: "
            f"{missing_preview}\nGT directory: {gt_path}"
        )

    return pairs


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = tf.to_tensor(image.convert("RGB")).unsqueeze(0)[:, :3, :, :]
    return tensor.to(device=device, non_blocking=True)


def evaluate_pairs(
    pairs: Sequence[Tuple[Path, Path, str]],
    device: torch.device,
    compute_lpips: bool,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    psnr_values: List[torch.Tensor] = []
    ssim_values: List[torch.Tensor] = []
    lpips_values: List[torch.Tensor] = []
    per_image: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for input_image, gt_image, image_name in tqdm(pairs, desc="Ablation metrics"):
            prediction = load_rgb_tensor(input_image, device)
            target = load_rgb_tensor(gt_image, device)
            if tuple(prediction.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Image shape mismatch for {image_name}: "
                    f"input={tuple(prediction.shape)}, GT={tuple(target.shape)}"
                )

            current_psnr = psnr(prediction, target).mean()
            current_ssim = ssim(prediction, target).mean()
            psnr_values.append(current_psnr.detach().cpu())
            ssim_values.append(current_ssim.detach().cpu())

            image_metrics = {
                "PSNR": float(current_psnr.item()),
                "SSIM": float(current_ssim.item()),
            }

            if compute_lpips:
                current_lpips = lpips(prediction, target, net_type="vgg").mean()
                lpips_values.append(current_lpips.detach().cpu())
                image_metrics["LPIPS"] = float(current_lpips.item())

            per_image[image_name] = image_metrics

    results = {
        "PSNR": float(torch.stack(psnr_values).mean().item()),
        "SSIM": float(torch.stack(ssim_values).mean().item()),
    }
    if compute_lpips:
        results["LPIPS"] = float(torch.stack(lpips_values).mean().item())

    return results, per_image


def stage_images_for_fid(image_paths: Sequence[Path], output_dir: Path) -> None:
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


def calculate_fid_for_pairs(
    pairs: Sequence[Tuple[Path, Path, str]],
    batch_size: int,
) -> float:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise ImportError(
            "FID computation requires the PyPI package 'pytorch-fid' "
            "(import name: pytorch_fid). Install it with: python -m pip install pytorch-fid"
        ) from exc

    if batch_size <= 0:
        raise ValueError(f"--fid_batch_size must be positive, got {batch_size}")

    input_paths = [pair[0] for pair in pairs]
    gt_paths = [pair[1] for pair in pairs]
    fid_device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory(prefix="mod_cob_gs_ablation_fid_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_subset = tmp_root / "input"
        gt_subset = tmp_root / "gt"
        stage_images_for_fid(input_paths, input_subset)
        stage_images_for_fid(gt_paths, gt_subset)
        return float(
            calculate_fid_given_paths(
                [str(gt_subset), str(input_subset)],
                batch_size,
                fid_device,
                2048,
                8,
            )
        )


def print_results(
    input_path: Path,
    gt_path: Path,
    pairs: Sequence[Tuple[Path, Path, str]],
    results: Dict[str, float],
    per_image: Dict[str, Dict[str, float]],
    show_per_image: bool,
) -> None:
    print("")
    print(f"Input: {input_path}")
    print(f"GT   : {gt_path}")
    print(f"Pairs: {len(pairs)}")
    print("")
    print("Mean metrics")
    print("  PSNR : {:>12.7f}".format(results["PSNR"]))
    print("  SSIM : {:>12.7f}".format(results["SSIM"]))
    if "LPIPS" in results:
        print("  LPIPS: {:>12.7f}".format(results["LPIPS"]))
    if "FID" in results:
        print("  FID  : {:>12.7f}".format(results["FID"]))

    if show_per_image:
        print("")
        metric_names = ["PSNR", "SSIM"] + (["LPIPS"] if "LPIPS" in results else [])
        print("Per-image metrics")
        print("image\t" + "\t".join(metric_names))
        for image_name in sorted(per_image):
            values = [f"{per_image[image_name][metric]:.7f}" for metric in metric_names]
            print(image_name + "\t" + "\t".join(values))


def main() -> None:
    args = parse_args()
    input_path = require_dir(args.input_path, "input_path")
    gt_path = require_dir(args.gt_path, "GT_path")
    device = torch.device(args.device)
    pairs = build_pairs(input_path, gt_path)
    results, per_image = evaluate_pairs(
        pairs=pairs,
        device=device,
        compute_lpips=not bool(args.skip_lpips),
    )
    if bool(args.compute_fid):
        results["FID"] = calculate_fid_for_pairs(pairs, int(args.fid_batch_size))
    print_results(
        input_path=input_path,
        gt_path=gt_path,
        pairs=pairs,
        results=results,
        per_image=per_image,
        show_per_image=bool(args.per_image),
    )


if __name__ == "__main__":
    main()
