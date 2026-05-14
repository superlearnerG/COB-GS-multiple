import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw .npy depth maps and remove pixels covered by the "
            "union of non-background object-mask labels."
        )
    )
    parser.add_argument(
        "--source_path",
        "--source-path",
        "-s",
        required=True,
        help="Dataset root. Defaults read depth from <source_path>/depth and masks from <source_path>/object_mask.",
    )
    parser.add_argument(
        "--raw_depth_dir",
        "--raw-depth-dir",
        default=None,
        help=(
            "Directory containing raw .npy depth maps. Default auto-detects "
            "<source_path>/depth first, then <source_path>/depth/raw_depth."
        ),
    )
    parser.add_argument(
        "--mask_dir",
        "--mask-dir",
        default=None,
        help="Directory containing grayscale object masks. Default: <source_path>/object_mask.",
    )
    parser.add_argument(
        "--depth_vis_dir",
        "--depth-vis-dir",
        default=None,
        help="Output directory for original depth visualization PNGs. Default: <source_path>/depth/depth_vis.",
    )
    parser.add_argument(
        "--depth_remove_dir",
        "--depth-remove-dir",
        default=None,
        help="Output directory for object-removed depth visualization PNGs. Default: <source_path>/depth/depth_remove.",
    )
    parser.add_argument(
        "--mask_extension",
        "--mask-extension",
        default=".png",
        help="Mask filename extension used with each raw depth basename. Default: .png.",
    )
    parser.add_argument(
        "--removed_value",
        "--removed-value",
        type=int,
        default=0,
        help="RGB value assigned to pixels covered by the union object mask. Default: 0.",
    )
    parser.add_argument(
        "--skip_missing_masks",
        "--skip-missing-masks",
        action="store_true",
        default=False,
        help="Skip depth maps whose matching object mask is missing instead of raising an error.",
    )
    parser.add_argument(
        "--no_resize_mask",
        "--no-resize-mask",
        action="store_true",
        default=False,
        help="Raise an error if an object mask resolution differs from the raw depth resolution.",
    )
    return parser.parse_args()


def require_dir(path: Path, name: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    return path


def has_npy_files(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.npy"))


def resolve_raw_depth_dir(source_path: Path, raw_depth_dir: str = None) -> Path:
    if raw_depth_dir is not None:
        return require_dir(Path(raw_depth_dir).expanduser().resolve(), "raw_depth_dir")

    depth_root = source_path / "depth"
    raw_depth_root = depth_root / "raw_depth"
    if has_npy_files(depth_root):
        return depth_root
    if has_npy_files(raw_depth_root):
        return raw_depth_root
    if raw_depth_root.is_dir():
        return raw_depth_root
    return require_dir(depth_root, "raw depth")


def list_raw_depth_files(raw_depth_dir: Path) -> List[Path]:
    depth_paths = sorted(raw_depth_dir.glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No .npy raw depth maps found under {raw_depth_dir}")
    return depth_paths


def normalize_extension(extension: str) -> str:
    extension = str(extension)
    return extension if extension.startswith(".") else f".{extension}"


def load_raw_depth(depth_path: Path) -> np.ndarray:
    depth = np.asarray(np.load(depth_path))
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            raise ValueError(f"Expected a single-channel raw depth map at {depth_path}, got shape {depth.shape}.")
    elif depth.ndim != 2:
        squeezed = np.squeeze(depth)
        if squeezed.ndim != 2:
            raise ValueError(f"Expected a 2D raw depth map at {depth_path}, got shape {depth.shape}.")
        depth = squeezed
    return depth.astype(np.float32, copy=False)


def encode_depth_for_visualization(depth: np.ndarray) -> Tuple[np.ndarray, float, float]:
    finite_mask = np.isfinite(depth)
    if np.any(finite_mask):
        depth_min = float(np.min(depth[finite_mask]))
        depth_max = float(np.max(depth[finite_mask]))
    else:
        depth_min = 0.0
        depth_max = 0.0

    depth_norm = np.zeros_like(depth, dtype=np.float32)
    denom = depth_max - depth_min
    if denom > EPS:
        depth_norm[finite_mask] = np.clip((depth[finite_mask] - depth_min) / (denom + EPS), 0.0, 1.0)

    depth_vis = np.stack(
        [
            0.25 * depth_norm,
            0.65 * depth_norm,
            depth_norm,
        ],
        axis=-1,
    )
    depth_vis = np.clip(depth_vis * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return depth_vis, depth_min, depth_max


def load_union_object_mask(mask_path: Path, target_shape: Tuple[int, int], resize_mask: bool) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask_image = image.convert("L")
        if mask_image.size != (target_shape[1], target_shape[0]):
            if not resize_mask:
                raise ValueError(
                    f"Mask shape mismatch for {mask_path}: mask={mask_image.size[::-1]}, depth={target_shape}"
                )
            mask_image = mask_image.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
        mask = np.asarray(mask_image)
    return (mask != 0) & (mask != 255)


def save_rgb_png(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output_path)


def process_depth_maps(
    depth_paths: Iterable[Path],
    mask_dir: Path,
    depth_vis_dir: Path,
    depth_remove_dir: Path,
    mask_extension: str,
    removed_value: int,
    skip_missing_masks: bool,
    resize_mask: bool,
) -> Tuple[int, int]:
    processed = 0
    skipped = 0
    removed_value = int(np.clip(removed_value, 0, 255))

    depth_vis_dir.mkdir(parents=True, exist_ok=True)
    depth_remove_dir.mkdir(parents=True, exist_ok=True)

    for depth_path in depth_paths:
        output_name = f"{depth_path.stem}.png"
        mask_path = mask_dir / f"{depth_path.stem}{mask_extension}"
        if not mask_path.exists():
            if skip_missing_masks:
                skipped += 1
                print(f"[skip] Missing mask for {depth_path.name}: expected {mask_path}")
                continue
            raise FileNotFoundError(f"Missing object mask for {depth_path.name}: expected {mask_path}")

        raw_depth = load_raw_depth(depth_path)
        depth_vis, depth_min, depth_max = encode_depth_for_visualization(raw_depth)
        union_mask = load_union_object_mask(mask_path, raw_depth.shape, resize_mask=resize_mask)

        depth_remove = depth_vis.copy()
        depth_remove[union_mask] = removed_value

        save_rgb_png(depth_vis, depth_vis_dir / output_name)
        save_rgb_png(depth_remove, depth_remove_dir / output_name)
        processed += 1
        print(
            f"[depth_remove] {depth_path.name} -> {output_name} "
            f"depth_min={depth_min:.6g} depth_max={depth_max:.6g} mask_pixels={int(union_mask.sum())}"
        )

    return processed, skipped


def main() -> None:
    args = parse_args()
    source_path = require_dir(Path(args.source_path).expanduser().resolve(), "source_path")
    raw_depth_dir = resolve_raw_depth_dir(source_path, args.raw_depth_dir)
    mask_dir = require_dir(
        Path(args.mask_dir).expanduser().resolve() if args.mask_dir is not None else source_path / "object_mask",
        "mask_dir",
    )
    depth_root = source_path / "depth"
    depth_vis_dir = (
        Path(args.depth_vis_dir).expanduser().resolve()
        if args.depth_vis_dir is not None
        else depth_root / "depth_vis"
    )
    depth_remove_dir = (
        Path(args.depth_remove_dir).expanduser().resolve()
        if args.depth_remove_dir is not None
        else depth_root / "depth_remove"
    )

    depth_paths = list_raw_depth_files(raw_depth_dir)
    processed, skipped = process_depth_maps(
        depth_paths=depth_paths,
        mask_dir=mask_dir,
        depth_vis_dir=depth_vis_dir,
        depth_remove_dir=depth_remove_dir,
        mask_extension=normalize_extension(args.mask_extension),
        removed_value=args.removed_value,
        skip_missing_masks=bool(args.skip_missing_masks),
        resize_mask=not bool(args.no_resize_mask),
    )
    print(
        f"[depth_remove] Done. processed={processed} skipped={skipped} "
        f"depth_vis_dir={depth_vis_dir} depth_remove_dir={depth_remove_dir}"
    )


if __name__ == "__main__":
    main()
