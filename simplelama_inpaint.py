import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SimpleLaMa inpainting for matching image and inpaint-mask "
            "directories."
        )
    )
    parser.add_argument(
        "--image_dir",
        "--image-dir",
        default=None,
        help=(
            "Directory containing RGB images that need inpainting. Required in "
            "ablation mode because the SimpleLaMA input images are specified manually."
        ),
    )
    parser.add_argument(
        "--mask_dir",
        "--mask-dir",
        default=None,
        help="Directory containing inpaint masks. By default names must match images.",
    )
    parser.add_argument(
        "--mask_path",
        "--mask-path",
        default=None,
        help=(
            "Mask path used with --use_own_mask_path. It may be a directory of "
            "per-image masks or one mask image shared by all inputs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="Directory where inpainted images are written.",
    )
    parser.add_argument(
        "--source_path",
        "--source-path",
        "-s",
        default=None,
        help=(
            "Dataset root for ablation mode. The script reads grayscale masks "
            "from <source_path>/object_mask."
        ),
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        "-m",
        default=None,
        help=(
            "Model root for ablation mode. Outputs are written under "
            "<model_path>/ablation/2d_inpainting_result_<ids>."
        ),
    )
    parser.add_argument(
        "--images",
        default="images",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inpainting_mask",
        "--inpainting-mask",
        "--inpainting_mask_id",
        "--inpainting-mask-id",
        "--inpainting_mask_ids",
        "--inpainting-mask-ids",
        dest="inpainting_mask_ids",
        nargs="+",
        default=None,
        help=(
            "Grayscale object ids from <source_path>/object_mask whose union becomes "
            "the pure-white inpainting mask. Supports space- or comma-separated ids."
        ),
    )
    parser.add_argument(
        "--use_own_mask_path",
        "--use-own-mask-path",
        action="store_true",
        default=False,
        help=(
            "Use masks from --mask_path instead of building a union from "
            "<source_path>/object_mask and --inpainting_mask ids."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to simple_lama_inpainting.SimpleLama, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--mask_threshold",
        "--mask-threshold",
        type=int,
        default=0,
        help=(
            "Mask pixels greater than this grayscale threshold are inpainted. "
            "Ignored when --valid_greyvalues is set. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--valid_greyvalues",
        "--valid-greyvalues",
        "--valid_greyvalue",
        "--valid-greyvalue",
        dest="valid_greyvalues",
        nargs="+",
        default=None,
        help=(
            "Optional exact grayscale values used as inpaint regions. Supports "
            "space-separated values, e.g. 1 2 3, or comma-separated values, e.g. 1,2,3."
        ),
    )
    parser.add_argument(
        "--mask_dilation",
        "--mask-dilation",
        type=int,
        default=0,
        help="Square dilation radius in pixels applied to the binary mask. Defaults to 0.",
    )
    parser.add_argument(
        "--image_extensions",
        "--image-extensions",
        nargs="+",
        default=None,
        help=(
            "Image filename extensions to process. Defaults to common image "
            "extensions. Values may be space-separated or comma-separated."
        ),
    )
    parser.add_argument(
        "--match_by_stem",
        "--match-by-stem",
        action="store_true",
        help=(
            "If an exact mask filename is missing, match one mask with the same "
            "stem and any supported image extension."
        ),
    )
    parser.add_argument(
        "--output_ext",
        "--output-ext",
        default=None,
        help=(
            "Optional output extension such as .png. Defaults to preserving the "
            "input image filename."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        "--skip-existing",
        action="store_true",
        help="Skip pairs whose output file already exists.",
    )
    return parser.parse_args()


def parse_extensions(raw_extensions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if raw_extensions is None:
        return DEFAULT_IMAGE_EXTENSIONS

    extensions: List[str] = []
    for raw_value in raw_extensions:
        for token in raw_value.split(","):
            token = token.strip().lower()
            if not token:
                continue
            if not token.startswith("."):
                token = f".{token}"
            extensions.append(token)

    if not extensions:
        raise ValueError("--image_extensions must contain at least one extension")

    return tuple(sorted(set(extensions)))


def parse_valid_greyvalues(raw_values: Optional[Sequence[str]]) -> Optional[Tuple[int, ...]]:
    if raw_values is None:
        return None

    greyvalues: List[int] = []
    for raw_value in raw_values:
        for token in raw_value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                greyvalue = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"valid_greyvalues must contain integers, got {token!r}"
                ) from exc
            if greyvalue < 0 or greyvalue > 255:
                raise ValueError(
                    f"valid_greyvalues must be in [0, 255], got {greyvalue}"
                )
            greyvalues.append(greyvalue)

    if not greyvalues:
        raise ValueError("--valid_greyvalues must contain at least one gray value")

    return tuple(sorted(set(greyvalues)))


def parse_id_values(raw_values: Optional[Sequence[str]], arg_name: str) -> Optional[Tuple[int, ...]]:
    if raw_values is None:
        return None

    ids: List[int] = []
    seen = set()
    for raw_value in raw_values:
        for token in raw_value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"{arg_name} must contain integers, got {token!r}") from exc
            if value < 0 or value > 255:
                raise ValueError(f"{arg_name} must be in [0, 255], got {value}")
            if value in seen:
                continue
            ids.append(value)
            seen.add(value)

    if not ids:
        raise ValueError(f"{arg_name} must contain at least one gray value")
    return tuple(ids)


def require_existing_dir(path: str, desc: str) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_dir():
        raise FileNotFoundError(f"{desc} directory not found: {resolved_path}")
    return resolved_path


def require_existing_path(path: str, desc: str) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{desc} path not found: {resolved_path}")
    if not resolved_path.is_file() and not resolved_path.is_dir():
        raise FileNotFoundError(f"{desc} must be a file or directory: {resolved_path}")
    return resolved_path


def iter_image_paths(image_dir: Path, image_extensions: Sequence[str]) -> List[Path]:
    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]
    return sorted(image_paths, key=lambda path: path.name)


def build_stem_index(
    mask_dir: Path,
    image_extensions: Sequence[str],
) -> Dict[str, List[Path]]:
    stem_index: Dict[str, List[Path]] = {}
    for path in mask_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in image_extensions:
            continue
        stem_index.setdefault(path.stem, []).append(path)
    return stem_index


def resolve_mask_path(
    image_path: Path,
    mask_dir: Path,
    match_by_stem: bool,
    stem_index: Dict[str, List[Path]],
) -> Path:
    exact_path = mask_dir / image_path.name
    if exact_path.is_file():
        return exact_path

    if not match_by_stem:
        raise FileNotFoundError(
            f"Mask not found for image {image_path.name}: expected {exact_path}"
        )

    candidates = sorted(stem_index.get(image_path.stem, []), key=lambda path: path.name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Mask not found for image {image_path.name}: no file with stem "
            f"{image_path.stem!r} in {mask_dir}"
        )

    candidate_names = ", ".join(path.name for path in candidates)
    raise ValueError(
        f"Ambiguous masks for image {image_path.name}: {candidate_names}. "
        "Use exact filenames or remove duplicates."
    )


def normalize_output_ext(output_ext: Optional[str]) -> Optional[str]:
    if output_ext is None:
        return None
    output_ext = output_ext.strip()
    if not output_ext:
        raise ValueError("--output_ext must not be empty")
    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"
    return output_ext


def build_output_path(image_path: Path, output_dir: Path, output_ext: Optional[str]) -> Path:
    if output_ext is None:
        return output_dir / image_path.name
    return output_dir / f"{image_path.stem}{output_ext}"


def format_id_suffix(ids: Sequence[int]) -> str:
    if not ids:
        raise ValueError("At least one inpainting mask id is required to build the output suffix")
    return "_".join(str(int(value)) for value in ids)


def format_path_suffix(path: Path) -> str:
    raw_name = path.stem if path.is_file() else path.name
    raw_name = raw_name.strip() or "own_mask"
    suffix = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw_name)
    return suffix.strip("._") or "own_mask"


def resolve_own_mask_path(args: argparse.Namespace) -> Path:
    if args.mask_path is None and args.mask_dir is None:
        raise ValueError("--mask_path is required when using --use_own_mask_path")
    if args.mask_path is not None and args.mask_dir is not None:
        mask_path = Path(args.mask_path).expanduser().resolve()
        mask_dir = Path(args.mask_dir).expanduser().resolve()
        if mask_path != mask_dir:
            raise ValueError("--mask_path and --mask_dir both refer to masks but point to different paths")
    return require_existing_path(args.mask_path if args.mask_path is not None else args.mask_dir, "mask_path")


def resolve_runtime_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Optional[Tuple[int, ...]], bool, bool]:
    inpainting_mask_ids = parse_id_values(args.inpainting_mask_ids, "--inpainting_mask_id")
    own_mask_mode = bool(args.use_own_mask_path)
    if own_mask_mode and inpainting_mask_ids is not None:
        raise ValueError("--inpainting_mask_id / --inpainting_mask cannot be used with --use_own_mask_path")
    if own_mask_mode and args.valid_greyvalues is not None:
        raise ValueError("--valid_greyvalues cannot be used with --use_own_mask_path")

    ablation_mode = any(
        value is not None
        for value in (args.source_path, args.model_path, args.inpainting_mask_ids)
    )

    if ablation_mode:
        if args.source_path is None and not own_mask_mode:
            raise ValueError("--source_path / -s is required when using --inpainting_mask_id")
        if args.model_path is None:
            raise ValueError("--model_path / -m is required in ablation mode")
        if inpainting_mask_ids is None and not own_mask_mode:
            raise ValueError("--inpainting_mask_id is required in ablation mode")
        if args.image_dir is None:
            raise ValueError("--image_dir is required in ablation mode")

        source_path = require_existing_dir(args.source_path, "source_path") if args.source_path is not None else None
        model_path = Path(args.model_path).expanduser().resolve()
        image_dir = require_existing_dir(args.image_dir, "image_dir")
        if own_mask_mode:
            mask_dir = resolve_own_mask_path(args)
            output_suffix = f"own_mask_{format_path_suffix(mask_dir)}"
        else:
            mask_dir = require_existing_dir(
                args.mask_dir if args.mask_dir is not None else str(source_path / "object_mask"),
                "object_mask",
            )
            output_suffix = format_id_suffix(inpainting_mask_ids)
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir is not None
            else model_path / "ablation" / f"2d_inpainting_result_{output_suffix}"
        )
        return image_dir, mask_dir, output_dir, inpainting_mask_ids, True, own_mask_mode

    if own_mask_mode:
        if args.image_dir is None or args.output_dir is None:
            raise ValueError("--image_dir and --output_dir are required when using --use_own_mask_path outside ablation mode")
        return (
            require_existing_dir(args.image_dir, "image_dir"),
            resolve_own_mask_path(args),
            Path(args.output_dir).expanduser().resolve(),
            None,
            False,
            True,
        )

    if args.image_dir is None or args.mask_dir is None or args.output_dir is None:
        raise ValueError(
            "Either use ablation mode with -s/--source_path, -m/--model_path, "
            "and --inpainting_mask_id, or provide --image_dir, --mask_dir, and --output_dir."
        )

    return (
        require_existing_dir(args.image_dir, "image_dir"),
        require_existing_dir(args.mask_dir, "mask_dir"),
        Path(args.output_dir).expanduser().resolve(),
        None,
        False,
        False,
    )


def build_pairs(
    image_dir: Path,
    mask_root: Path,
    output_dir: Path,
    image_extensions: Sequence[str],
    match_by_stem: bool,
    output_ext: Optional[str],
) -> List[Tuple[Path, Path, Path]]:
    image_paths = iter_image_paths(image_dir, image_extensions)
    if not image_paths:
        raise FileNotFoundError(
            f"No input images with extensions {tuple(image_extensions)} in {image_dir}"
        )

    if mask_root.is_file():
        return [
            (image_path, mask_root, build_output_path(image_path, output_dir, output_ext))
            for image_path in image_paths
        ]

    stem_index = build_stem_index(mask_root, image_extensions) if match_by_stem else {}
    pairs: List[Tuple[Path, Path, Path]] = []
    for image_path in image_paths:
        resolved_mask_path = resolve_mask_path(
            image_path=image_path,
            mask_dir=mask_root,
            match_by_stem=match_by_stem,
            stem_index=stem_index,
        )
        output_path = build_output_path(image_path, output_dir, output_ext)
        pairs.append((image_path, resolved_mask_path, output_path))

    return pairs


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_binary_mask(
    path: Path,
    mask_threshold: int,
    valid_greyvalues: Optional[Sequence[int]],
    dilation_radius: int,
) -> Image.Image:
    if mask_threshold < 0 or mask_threshold > 255:
        raise ValueError(f"--mask_threshold must be in [0, 255], got {mask_threshold}")
    if dilation_radius < 0:
        raise ValueError(f"--mask_dilation must be >= 0, got {dilation_radius}")

    with Image.open(path) as mask:
        mask_np = np.asarray(mask.convert("L"), dtype=np.uint8)

    if valid_greyvalues is None:
        binary = np.where(mask_np > mask_threshold, 255, 0).astype(np.uint8)
    else:
        binary = np.where(np.isin(mask_np, valid_greyvalues), 255, 0).astype(np.uint8)

    mask_image = Image.fromarray(binary).convert("L")
    if dilation_radius > 0:
        filter_size = 2 * dilation_radius + 1
        mask_image = mask_image.filter(ImageFilter.MaxFilter(filter_size))

    return mask_image


def init_lama(device: str):
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise ImportError(
            "simple_lama_inpainting is required. Install it in the environment "
            "used to run this script."
        ) from exc

    return SimpleLama(device=device)


def crop_to_input_size(result: Image.Image, input_size: Tuple[int, int]) -> Image.Image:
    if result.size == input_size:
        return result
    width, height = input_size
    return result.crop((0, 0, width, height))


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
    image.save(output_path)


def run_batch(
    image_dir: Path,
    mask_path: Path,
    output_dir: Path,
    device: str,
    mask_threshold: int,
    valid_greyvalues: Optional[Sequence[int]],
    mask_dilation: int,
    image_extensions: Sequence[str],
    match_by_stem: bool,
    output_ext: Optional[str],
    skip_existing: bool,
) -> None:
    image_dir_path = image_dir
    mask_path_value = mask_path
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(
        image_dir=image_dir_path,
        mask_root=mask_path_value,
        output_dir=output_dir,
        image_extensions=image_extensions,
        match_by_stem=match_by_stem,
        output_ext=output_ext,
    )

    simple_lama = init_lama(device=device)
    total = len(pairs)
    for index, (image_path, mask_path, output_path) in enumerate(pairs, start=1):
        if skip_existing and output_path.is_file():
            print(f"[SimpleLaMa] [{index}/{total}] skip existing: {output_path}")
            continue

        image = load_rgb_image(image_path)
        mask = load_binary_mask(
            path=mask_path,
            mask_threshold=mask_threshold,
            valid_greyvalues=valid_greyvalues,
            dilation_radius=mask_dilation,
        )
        if image.size != mask.size:
            raise RuntimeError(
                "Image/mask size mismatch: "
                f"{image_path.name} size={image.size}, "
                f"{mask_path.name} size={mask.size}"
            )

        result = simple_lama(image, mask)
        result = crop_to_input_size(result, image.size)
        save_image(result, output_path)
        print(
            f"[SimpleLaMa] [{index}/{total}] "
            f"{image_path.name} + {mask_path.name} -> {output_path}"
        )


def main() -> None:
    args = parse_args()
    image_dir, mask_path, output_dir, inpainting_mask_ids, ablation_mode, own_mask_mode = resolve_runtime_paths(args)
    valid_greyvalues = None if own_mask_mode else inpainting_mask_ids
    if valid_greyvalues is None and not own_mask_mode:
        valid_greyvalues = parse_valid_greyvalues(args.valid_greyvalues)
    match_by_stem = bool(args.match_by_stem or ablation_mode)
    if ablation_mode:
        print(f"[SimpleLaMa] ablation image_dir: {image_dir}")
        if own_mask_mode:
            print(f"[SimpleLaMa] ablation own mask_path: {mask_path}")
            print("[SimpleLaMa] ablation mask mode: threshold")
        else:
            print(f"[SimpleLaMa] ablation object_mask: {mask_path}")
            print(f"[SimpleLaMa] ablation inpainting ids: {format_id_suffix(inpainting_mask_ids)}")
        print(f"[SimpleLaMa] ablation output_dir: {output_dir}")
    run_batch(
        image_dir=image_dir,
        mask_path=mask_path,
        output_dir=output_dir,
        device=args.device,
        mask_threshold=args.mask_threshold,
        valid_greyvalues=valid_greyvalues,
        mask_dilation=args.mask_dilation,
        image_extensions=parse_extensions(args.image_extensions),
        match_by_stem=match_by_stem,
        output_ext=normalize_output_ext(args.output_ext),
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
