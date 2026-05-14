from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GROUPING_ROOT = REPO_ROOT.parent.parent
SHARED_TORCH_HOME_CANDIDATES = [
    GROUPING_ROOT / "compare_methods" / "pretrained_models" / "torch",
    REPO_ROOT.parent / "pretrained_models" / "torch",
]
DEFAULT_TORCH_HOME = next(
    (path for path in SHARED_TORCH_HOME_CANDIDATES if path.exists()),
    SHARED_TORCH_HOME_CANDIDATES[0],
)
os.environ["TORCH_HOME"] = str(DEFAULT_TORCH_HOME)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


FID_DIMS = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR/SSIM/LPIPS/FID for one GT image and one prediction image. "
            "When --mask is provided, also compute PSNR_masked/SSIM_masked/"
            "LPIPS_masked/FID_masked."
        )
    )
    parser.add_argument("--gt", required=True, help="Ground-truth image path.")
    parser.add_argument(
        "--pred",
        "--prediction",
        required=True,
        dest="pred",
        help="Prediction/render image path.",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help=(
            "Optional mask image path for masked metrics. By default, nonzero "
            "mask pixels are evaluated."
        ),
    )
    parser.add_argument(
        "--mask-values",
        default=None,
        help="Comma-separated gray values to keep, for example '255' or '12,25,255'.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=0,
        help="Pixels greater than this value are kept when --mask-values is not set.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Metric device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--skip-fid",
        action="store_true",
        help="Skip FID/FID_masked. Useful for quick checks on CPU.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=4,
        help="Batch size for Inception feature extraction. Defaults to 4.",
    )
    parser.add_argument(
        "--allow-fid-download",
        action="store_true",
        help=(
            "Allow pytorch-fid to download the Inception checkpoint if no local "
            "checkpoint is found."
        ),
    )
    parser.add_argument(
        "--resize-pred-to-gt",
        action="store_true",
        help=(
            "Resize the prediction to the GT resolution with bicubic interpolation "
            "when image sizes differ. Without this flag, size mismatch is an error."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the metrics as JSON.",
    )
    return parser.parse_args()


def require_file(path: str, desc: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{desc} not found: {resolved}")
    return resolved


def parse_mask_values(raw_values: str | None) -> list[int] | None:
    if raw_values is None:
        return None

    values: list[int] = []
    for token in raw_values.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value > 255:
            raise ValueError(f"Mask gray value must be in [0, 255], got {value}")
        values.append(value)

    if not values:
        raise ValueError("--mask-values was set but no values were parsed")
    return values


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)
    return tensor.to(device=device, non_blocking=True)


def load_image_pair(
    gt_path: Path,
    pred_path: Path,
    resize_pred_to_gt: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Image.Image, Image.Image]:
    gt_image = load_rgb_image(gt_path)
    pred_image = load_rgb_image(pred_path)

    if pred_image.size != gt_image.size:
        if not resize_pred_to_gt:
            raise RuntimeError(
                "Image size mismatch: "
                f"GT={gt_image.size} prediction={pred_image.size}. "
                "Pass --resize-pred-to-gt if this resize is intentional."
            )
        pred_image = pred_image.resize(gt_image.size, Image.Resampling.BICUBIC)

    gt_tensor = image_to_tensor(gt_image, device)
    pred_tensor = image_to_tensor(pred_image, device)
    return gt_tensor, pred_tensor, gt_image, pred_image


def load_mask_tensor(
    mask_path: Path,
    target_size: tuple[int, int],
    mask_values: list[int] | None,
    threshold: int,
    device: torch.device,
) -> torch.Tensor:
    with Image.open(mask_path) as image:
        mask_image = image.convert("L")
        if mask_image.size != target_size:
            mask_image = mask_image.resize(target_size, Image.Resampling.NEAREST)
        mask_np = np.asarray(mask_image)

    if mask_values is None:
        mask = mask_np > threshold
    else:
        mask = np.isin(mask_np, np.asarray(mask_values, dtype=np.uint8))

    if not mask.any():
        selector = f">{threshold}" if mask_values is None else str(mask_values)
        raise RuntimeError(f"Mask {mask_path} has no selected pixels: {selector}")

    return (
        torch.from_numpy(mask.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, non_blocking=True)
    )


def compute_masked_psnr(
    pred_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
) -> float:
    mask_pixels = mask_tensor.sum()
    if mask_pixels.item() <= 0:
        raise RuntimeError("Cannot compute masked PSNR with an empty mask")

    mse = (((pred_tensor - gt_tensor) ** 2) * mask_tensor).sum()
    mse = mse / (mask_pixels * pred_tensor.shape[1])
    if mse.item() <= 0:
        return float("inf")
    return float((20 * torch.log10(1.0 / torch.sqrt(mse))).item())


def crop_to_mask_bbox(tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
    indices = torch.nonzero(mask_tensor[0, 0] > 0, as_tuple=False)
    if indices.numel() == 0:
        raise RuntimeError("Cannot crop an empty mask")

    y0 = int(indices[:, 0].min().item())
    y1 = int(indices[:, 0].max().item()) + 1
    x0 = int(indices[:, 1].min().item())
    x1 = int(indices[:, 1].max().item()) + 1
    return tensor[..., y0:y1, x0:x1]


def masked_local_tensors(
    pred_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_bool = mask_tensor.bool()
    masked_pred = torch.where(mask_bool, pred_tensor, gt_tensor)
    return crop_to_mask_bbox(masked_pred, mask_tensor), crop_to_mask_bbox(gt_tensor, mask_tensor)


def configure_fid_cache(allow_download: bool) -> Path | None:
    try:
        from pytorch_fid.inception import FID_WEIGHTS_URL
    except ImportError as exc:
        raise ImportError(
            "FID requires the PyPI package 'pytorch-fid' "
            "(import name: pytorch_fid)."
        ) from exc

    weights_name = Path(urlparse(FID_WEIGHTS_URL).path).name
    candidate_hubs = [
        *(path / "hub" for path in SHARED_TORCH_HOME_CANDIDATES),
        Path(torch.hub.get_dir()).expanduser(),
        Path.home() / ".cache" / "torch" / "hub",
    ]

    seen: set[Path] = set()
    checked: list[Path] = []
    for hub_dir in candidate_hubs:
        hub_dir = hub_dir.expanduser().resolve()
        if hub_dir in seen:
            continue
        seen.add(hub_dir)
        checkpoint_path = hub_dir / "checkpoints" / weights_name
        checked.append(checkpoint_path)
        if checkpoint_path.is_file():
            torch.hub.set_dir(str(hub_dir))
            return checkpoint_path

    fallback_hub = (DEFAULT_TORCH_HOME / "hub").expanduser().resolve()
    torch.hub.set_dir(str(fallback_hub))
    if allow_download:
        return None

    raise FileNotFoundError(
        "pytorch-fid Inception checkpoint not found. Checked:\n"
        + "\n".join(f"  {path}" for path in checked)
        + "\nPass --allow-fid-download to let pytorch-fid download it."
    )


def inception_activations(
    tensors: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
    allow_download: bool,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError(f"--fid-batch-size must be positive, got {batch_size}")

    fid_checkpoint = configure_fid_cache(allow_download)
    if fid_checkpoint is not None:
        print(f"Using FID checkpoint: {fid_checkpoint}")

    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
    model = InceptionV3([block_idx]).to(device).eval()

    activations: list[np.ndarray] = []

    def flush_batch(batch: list[torch.Tensor]) -> None:
        images = torch.cat(batch, dim=0).to(device)
        pred = model(images)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        activations.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())

    current_batch: list[torch.Tensor] = []
    current_shape: tuple[int, ...] | None = None
    with torch.no_grad():
        for tensor in tensors:
            tensor_shape = tuple(tensor.shape[1:])
            if (
                current_batch
                and (tensor_shape != current_shape or len(current_batch) >= batch_size)
            ):
                flush_batch(current_batch)
                current_batch = []
            current_shape = tensor_shape
            current_batch.append(tensor)
        if current_batch:
            flush_batch(current_batch)

    return np.concatenate(activations, axis=0)


def fid_from_activations(act_a: np.ndarray, act_b: np.ndarray) -> float:
    mu_a = np.mean(act_a, axis=0)
    mu_b = np.mean(act_b, axis=0)

    if act_a.shape[0] < 2 and act_b.shape[0] < 2:
        diff = mu_a - mu_b
        return float(np.dot(diff, diff))

    from pytorch_fid.fid_score import calculate_frechet_distance

    sigma_a = np.cov(act_a, rowvar=False) if act_a.shape[0] > 1 else np.zeros((FID_DIMS, FID_DIMS))
    sigma_b = np.cov(act_b, rowvar=False) if act_b.shape[0] > 1 else np.zeros((FID_DIMS, FID_DIMS))
    return float(calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b))


def compute_fid_values(
    gt_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int,
    allow_download: bool,
    masked_gt_tensor: torch.Tensor | None = None,
    masked_pred_tensor: torch.Tensor | None = None,
) -> tuple[float, float | None]:
    tensors = [gt_tensor, pred_tensor]
    if masked_gt_tensor is not None and masked_pred_tensor is not None:
        tensors.extend([masked_gt_tensor, masked_pred_tensor])

    activations = inception_activations(tensors, device, batch_size, allow_download)
    fid = fid_from_activations(activations[0:1], activations[1:2])
    fid_masked = None
    if masked_gt_tensor is not None and masked_pred_tensor is not None:
        fid_masked = fid_from_activations(activations[2:3], activations[3:4])
    return fid, fid_masked


def format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.7f}"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return None
    return value


def main() -> None:
    args = parse_args()
    gt_path = require_file(args.gt, "GT image")
    pred_path = require_file(args.pred, "Prediction image")
    mask_path = require_file(args.mask, "Mask image") if args.mask else None
    mask_values = parse_mask_values(args.mask_values)
    device = torch.device(args.device)

    gt_tensor, pred_tensor, gt_image, _ = load_image_pair(
        gt_path,
        pred_path,
        args.resize_pred_to_gt,
        device,
    )

    lpips_model = LPIPS("vgg").to(device).eval()
    results: dict[str, float | int | str | list[int] | None] = {
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "width": int(gt_tensor.shape[-1]),
        "height": int(gt_tensor.shape[-2]),
    }

    with torch.no_grad():
        results["PSNR"] = float(psnr(pred_tensor, gt_tensor).mean().item())
        results["SSIM"] = float(ssim(pred_tensor, gt_tensor).mean().item())
        results["LPIPS"] = float(lpips_model(pred_tensor, gt_tensor).mean().item())

        masked_gt_tensor = None
        masked_pred_tensor = None
        if mask_path is not None:
            mask_tensor = load_mask_tensor(
                mask_path,
                gt_image.size,
                mask_values,
                args.mask_threshold,
                device,
            )
            masked_pred_tensor, masked_gt_tensor = masked_local_tensors(
                pred_tensor,
                gt_tensor,
                mask_tensor,
            )
            results["mask_path"] = str(mask_path)
            results["mask_values"] = mask_values
            results["mask_threshold"] = args.mask_threshold if mask_values is None else None
            results["mask_pixels"] = int(mask_tensor.sum().item())
            results["PSNR_masked"] = compute_masked_psnr(pred_tensor, gt_tensor, mask_tensor)
            results["SSIM_masked"] = float(ssim(masked_pred_tensor, masked_gt_tensor).mean().item())
            results["LPIPS_masked"] = float(
                lpips_model(masked_pred_tensor, masked_gt_tensor).mean().item()
            )

        if args.skip_fid:
            results["FID"] = None
            if mask_path is not None:
                results["FID_masked"] = None
        else:
            fid, fid_masked = compute_fid_values(
                gt_tensor,
                pred_tensor,
                device,
                args.fid_batch_size,
                args.allow_fid_download,
                masked_gt_tensor,
                masked_pred_tensor,
            )
            results["FID"] = fid
            if mask_path is not None:
                results["FID_masked"] = fid_masked

    print(f"GT        : {gt_path}")
    print(f"Prediction: {pred_path}")
    if mask_path is not None:
        print(f"Mask      : {mask_path}")
    print("")
    for key in ("PSNR", "SSIM", "LPIPS", "FID"):
        print(f"{key:<12}: {format_value(results.get(key))}")
    if mask_path is not None:
        print("")
        for key in ("PSNR_masked", "SSIM_masked", "LPIPS_masked", "FID_masked"):
            print(f"{key:<12}: {format_value(results.get(key))}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(json_safe(results), file, indent=2)
        print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
    main()
