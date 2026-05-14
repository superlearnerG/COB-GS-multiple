from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene.colmap_loader import (  # noqa: E402
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from utils.graphics_utils import focal2fov, fov2focal  # noqa: E402


DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class ProjectionCamera:
    uid: int
    R: np.ndarray
    T: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    image_name: str
    image_path: str
    width: int
    height: int
    is_test: bool = False

    def c2w(self) -> np.ndarray:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = self.R.transpose()
        w2c[:3, 3] = self.T
        return np.linalg.inv(w2c).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a completed desk atlas texture/support mask into dataset camera "
            "views and overlay the projected texture on removal views."
        )
    )
    parser.add_argument("--source_path", "--source-path", "-s", required=True, help="Dataset root.")
    parser.add_argument("--model_path", "--model-path", "-m", required=True, help="Model output root.")
    parser.add_argument(
        "--removal_view_path",
        "--removal-view-path",
        "--removal_dir",
        "--removal-dir",
        required=True,
        help="Directory containing removal-view RGB images to be covered by the projected atlas texture.",
    )
    parser.add_argument(
        "--desk_atlas_dir",
        "--desk-atlas-dir",
        default="desk_atlas",
        help="Desk atlas directory. Relative paths are resolved under --model_path.",
    )
    parser.add_argument(
        "--texture_name",
        "--texture-name",
        default="texture_completed.png",
        help="Completed texture image name inside --desk_atlas_dir.",
    )
    parser.add_argument(
        "--support_mask_name",
        "--support-mask-name",
        default="M_support_footprint.png",
        help="Binary support footprint mask name inside --desk_atlas_dir.",
    )
    parser.add_argument(
        "--mask_dilation",
        "--mask-dilation",
        type=int,
        default=0,
        help="Square dilation radius in atlas pixels applied to the support footprint mask.",
    )
    parser.add_argument(
        "--debug_dir",
        "--debug-dir",
        default="debug/desk_atlas_virtual_supervision",
        help="Debug output directory. Relative paths are resolved under --model_path.",
    )
    parser.add_argument(
        "--source_type",
        "--source-type",
        choices=("auto", "colmap", "blender"),
        default="auto",
        help="Dataset camera format.",
    )
    parser.add_argument(
        "--images",
        default="images",
        help="COLMAP image directory name relative to --source_path.",
    )
    parser.add_argument(
        "--camera_split",
        "--camera-split",
        choices=("all", "train", "test"),
        default="all",
        help="Camera subset to project. For COLMAP, train_list.txt/test_list.txt are used when available.",
    )
    parser.add_argument(
        "--blender_extension",
        "--blender-extension",
        default=".png",
        help="Image extension appended to Blender transform file paths without a suffix.",
    )
    parser.add_argument(
        "--white_background",
        "--white-background",
        action="store_true",
        help="Accepted for parity with training scripts; camera projection itself does not use image alpha.",
    )
    parser.add_argument(
        "--background_transparent",
        "--background-transparent",
        action="store_true",
        help=(
            "Save black-background atlas/projection debug images as RGBA PNGs, "
            "using their masks as alpha."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Projection device: auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--row_chunk",
        "--row-chunk",
        type=int,
        default=256,
        help="Number of camera rows projected per chunk.",
    )
    parser.add_argument(
        "--overlay_alpha",
        "--overlay-alpha",
        type=float,
        default=1.0,
        help="Overlay alpha for projected texture on removal views. 1.0 means hard replacement.",
    )
    parser.add_argument(
        "--recursive_removal_search",
        "--recursive-removal-search",
        action="store_true",
        help="Search removal-view images recursively under --removal_view_path.",
    )
    parser.add_argument(
        "--skip_missing_removal",
        "--skip-missing-removal",
        action="store_true",
        help="Deprecated. Missing removal views are skipped by default.",
    )
    return parser.parse_args()


def resolve_child(root: str, child: str) -> Path:
    child_path = Path(child)
    if child_path.is_absolute():
        return child_path
    return Path(root) / child_path


def torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_desk_atlas_state(state_path: Path, device: torch.device):
    if not state_path.exists():
        raise FileNotFoundError(f"Missing desk atlas state: {state_path}")
    payload = torch_load_cpu(state_path)
    plane_payload = payload["plane"]

    def tensor_value(name: str) -> torch.Tensor:
        return torch.as_tensor(plane_payload[name], dtype=torch.float32, device=device)

    plane = SimpleNamespace(
        normal=tensor_value("normal"),
        d=tensor_value("d"),
        origin=tensor_value("origin"),
        e1=tensor_value("e1"),
        e2=tensor_value("e2"),
    )
    return SimpleNamespace(
        plane=plane,
        uv_bbox=tuple(float(v) for v in payload["uv_bbox"]),
        atlas_hw=tuple(int(v) for v in payload["atlas_hw"]),
    )


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing texture image: {path}")
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)


def load_binary_mask_image(path: Path, dilation_radius: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing support mask: {path}")
    if dilation_radius < 0:
        raise ValueError(f"--mask_dilation must be >= 0, got {dilation_radius}")
    with Image.open(path) as image:
        mask_np = np.asarray(image.convert("L"), dtype=np.uint8)
    mask_image = Image.fromarray(np.where(mask_np > 127, 255, 0).astype(np.uint8), mode="L")
    if dilation_radius > 0:
        mask_image = mask_image.filter(ImageFilter.MaxFilter(2 * dilation_radius + 1))
    return mask_image


def mask_image_to_tensor(mask_image: Image.Image, size_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    target_h, target_w = size_hw
    if mask_image.size != (target_w, target_h):
        mask_image = mask_image.resize((target_w, target_h), Image.Resampling.NEAREST)
    mask_np = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device=device)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_np = np.asarray(rgb, dtype=np.uint8)
    alpha_np = np.asarray(alpha, dtype=np.uint8)
    if alpha_np.ndim != 2:
        raise ValueError(f"Alpha mask must be 2D, got shape {alpha_np.shape}")
    if rgb_np.shape[:2] != alpha_np.shape:
        raise ValueError(f"RGB/alpha size mismatch: rgb={rgb_np.shape[:2]} alpha={alpha_np.shape}")
    rgba = np.dstack([rgb_np, alpha_np])
    rgba[alpha_np == 0, :3] = 0
    Image.fromarray(rgba, mode="RGBA").save(path)


def save_mask(path: Path, mask: np.ndarray, background_transparent: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_np = np.asarray(mask, dtype=np.uint8)
    if background_transparent:
        mask_rgb = np.repeat(mask_np[:, :, None], 3, axis=2)
        save_rgba(path, mask_rgb, mask_np)
        return
    Image.fromarray(mask_np, mode="L").save(path)


def save_atlas_debug(
    debug_dir: Path,
    texture: torch.Tensor,
    raw_mask: torch.Tensor,
    dilated_mask: torch.Tensor,
    background_transparent: bool = False,
) -> None:
    raw_mask_np = (
        raw_mask.squeeze(0).squeeze(0).detach().cpu().numpy() > 0.5
    ).astype(np.uint8) * 255
    dilated_mask_np = (
        dilated_mask.squeeze(0).squeeze(0).detach().cpu().numpy() > 0.5
    ).astype(np.uint8) * 255
    texture_np = (
        texture.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0).numpy() * 255.0
    ).round().astype(np.uint8)
    masked_texture = texture_np.copy()
    masked_texture[dilated_mask_np == 0] = 0
    save_mask(
        debug_dir / "atlas_mask_before_dilation.png",
        raw_mask_np,
        background_transparent=background_transparent,
    )
    save_mask(
        debug_dir / "atlas_mask_dilated.png",
        dilated_mask_np,
        background_transparent=background_transparent,
    )
    if background_transparent:
        save_rgba(debug_dir / "atlas_texture_masked.png", masked_texture, dilated_mask_np)
    else:
        save_rgb(debug_dir / "atlas_texture_masked.png", masked_texture)


def split_name_keys(name: str) -> set:
    stripped = name.strip()
    if not stripped:
        return set()
    basename = os.path.basename(stripped)
    stem = Path(basename).stem
    return {stripped, basename, stem}


def read_split_list(list_path: Path) -> set:
    names = set()
    if not list_path.exists():
        return names
    with list_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            names.update(split_name_keys(line))
    return names


def name_in_split(name: str, split_names: set) -> bool:
    if not split_names:
        return False
    return bool(split_name_keys(name) & split_names)


def is_numbered_test_image(image_name: str) -> bool:
    stem = Path(image_name).stem
    return stem.isdigit() and int(stem) % 8 == 0


def parse_colmap_pinhole_intrinsics(intr) -> Tuple[float, float, float, float]:
    model = str(intr.model).upper()
    params = intr.params
    if model == "SIMPLE_PINHOLE":
        focal = float(params[0])
        return focal, focal, float(params[1]), float(params[2])
    if model == "PINHOLE":
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    raise ValueError(
        f"COLMAP camera model '{intr.model}' is not supported. Use PINHOLE or SIMPLE_PINHOLE."
    )


def load_colmap_cameras(source_path: Path, images_dir: str, camera_split: str) -> List[ProjectionCamera]:
    try:
        extrinsics = read_extrinsics_binary(str(source_path / "sparse/0/images.bin"))
        intrinsics = read_intrinsics_binary(str(source_path / "sparse/0/cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(str(source_path / "sparse/0/images.txt"))
        intrinsics = read_intrinsics_text(str(source_path / "sparse/0/cameras.txt"))

    test_names = read_split_list(source_path / "test_list.txt")
    train_names = read_split_list(source_path / "train_list.txt")
    if not test_names:
        for key in extrinsics:
            image_name = extrinsics[key].name
            if is_numbered_test_image(image_name):
                test_names.update(split_name_keys(image_name))

    cameras: List[ProjectionCamera] = []
    for idx, key in enumerate(extrinsics):
        extr = extrinsics[key]
        intr = intrinsics[extr.camera_id]
        fx, fy, cx, cy = parse_colmap_pinhole_intrinsics(intr)
        image_name = extr.name
        is_test = name_in_split(image_name, test_names)
        is_train = name_in_split(image_name, train_names) if train_names else not is_test
        if camera_split == "test" and not is_test:
            continue
        if camera_split == "train" and not is_train:
            continue
        cameras.append(
            ProjectionCamera(
                uid=int(intr.id),
                R=np.transpose(qvec2rotmat(extr.qvec)).astype(np.float32),
                T=np.asarray(extr.tvec, dtype=np.float32),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                image_name=image_name,
                image_path=str(source_path / images_dir / image_name),
                width=int(intr.width),
                height=int(intr.height),
                is_test=is_test,
            )
        )
    return sorted(cameras, key=lambda cam: cam.image_name)


def resolve_blender_image_path(source_path: Path, file_path: str, extension: str) -> Path:
    candidate = source_path / file_path
    if candidate.suffix and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    suffix = extension if extension.startswith(".") else f".{extension}"
    return source_path / f"{file_path}{suffix}"


def load_blender_transform_cameras(
    source_path: Path,
    transforms_file: str,
    extension: str,
    is_test: bool,
) -> List[ProjectionCamera]:
    path = source_path / transforms_file
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        contents = json.load(handle)
    fovx = float(contents["camera_angle_x"])
    cameras: List[ProjectionCamera] = []
    for idx, frame in enumerate(contents["frames"]):
        image_path = resolve_blender_image_path(source_path, frame["file_path"], extension)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing Blender image referenced by {path}: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size

        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3]).astype(np.float32)
        T = w2c[:3, 3].astype(np.float32)
        fovy = focal2fov(fov2focal(fovx, width), height)
        fx = float(fov2focal(fovx, width))
        fy = float(fov2focal(fovy, height))
        image_name = image_path.name
        cameras.append(
            ProjectionCamera(
                uid=idx,
                R=R,
                T=T,
                fx=fx,
                fy=fy,
                cx=float(width - 1) / 2.0,
                cy=float(height - 1) / 2.0,
                image_name=image_name,
                image_path=str(image_path),
                width=int(width),
                height=int(height),
                is_test=is_test,
            )
        )
    return cameras


def load_cameras(args: argparse.Namespace) -> List[ProjectionCamera]:
    source_path = Path(args.source_path)
    source_type = args.source_type
    if source_type == "auto":
        if (source_path / "sparse").exists():
            source_type = "colmap"
        elif (source_path / "transforms_train.json").exists() or (source_path / "transforms_test.json").exists():
            source_type = "blender"
        else:
            raise FileNotFoundError(f"Cannot detect camera format under {source_path}")

    if source_type == "colmap":
        cameras = load_colmap_cameras(source_path, args.images, args.camera_split)
    else:
        train_cameras = load_blender_transform_cameras(
            source_path, "transforms_train.json", args.blender_extension, is_test=False
        )
        test_cameras = load_blender_transform_cameras(
            source_path, "transforms_test.json", args.blender_extension, is_test=True
        )
        if args.camera_split == "train":
            cameras = train_cameras
        elif args.camera_split == "test":
            cameras = test_cameras
        else:
            cameras = train_cameras + test_cameras

    if not cameras:
        raise RuntimeError(f"No cameras loaded from {source_path} with camera_split={args.camera_split}")
    return cameras


def iter_image_files(root: Path, recursive: bool) -> Iterable[Path]:
    paths = root.rglob("*") if recursive else root.iterdir()
    for path in paths:
        if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            yield path


def build_removal_index(root: Path, recursive: bool) -> Dict[str, List[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Missing removal-view directory: {root}")
    index: Dict[str, List[Path]] = {}
    for path in iter_image_files(root, recursive=recursive):
        keys = {path.name, path.stem}
        try:
            keys.add(str(path.relative_to(root)))
        except ValueError:
            pass
        for key in keys:
            index.setdefault(key, []).append(path)
    return index


def resolve_removal_path(root: Path, index: Dict[str, List[Path]], camera: ProjectionCamera) -> Optional[Path]:
    direct_candidates = [
        root / camera.image_name,
        root / Path(camera.image_name).name,
    ]
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    keys = [camera.image_name, Path(camera.image_name).name, Path(camera.image_name).stem]
    matches: List[Path] = []
    for key in keys:
        matches.extend(index.get(key, []))
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        raise RuntimeError(
            f"Ambiguous removal views for camera {camera.image_name}: "
            + ", ".join(str(path) for path in unique_matches)
        )
    return None


def debug_name(camera: ProjectionCamera) -> str:
    normalized = camera.image_name.replace("\\", "/").replace("/", "__")
    stem = Path(normalized).stem
    if not stem:
        stem = f"camera_{camera.uid:04d}"
    return f"{stem}.png"


def project_atlas_to_camera(
    camera: ProjectionCamera,
    state,
    texture: torch.Tensor,
    atlas_mask_raw: torch.Tensor,
    atlas_mask: torch.Tensor,
    device: torch.device,
    row_chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, atlas_h, atlas_w = texture.shape
    height = int(camera.height)
    width = int(camera.width)
    row_chunk = max(int(row_chunk), 1)

    projected_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    projected_mask_raw = np.zeros((height, width), dtype=np.uint8)
    projected_mask = np.zeros((height, width), dtype=np.uint8)

    c2w = torch.from_numpy(camera.c2w()).to(device=device, dtype=torch.float32)
    cam_origin = c2w[:3, 3]
    plane = state.plane
    u_min, u_max, v_min, v_max = state.uv_bbox
    du = max(float(u_max - u_min), 1e-6)
    dv = max(float(v_max - v_min), 1e-6)
    x_den = max(atlas_w - 1, 1)
    y_den = max(atlas_h - 1, 1)

    xs = torch.arange(width, dtype=torch.float32, device=device)
    for row_start in range(0, height, row_chunk):
        row_end = min(row_start + row_chunk, height)
        ys = torch.arange(row_start, row_end, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        dirs_cam = torch.stack(
            [
                (xx - float(camera.cx)) / float(camera.fx),
                (yy - float(camera.cy)) / float(camera.fy),
                torch.ones_like(xx),
            ],
            dim=-1,
        )
        dirs_cam = F.normalize(dirs_cam, dim=-1)
        dirs_world = torch.matmul(dirs_cam, c2w[:3, :3].transpose(0, 1))

        denom = torch.sum(dirs_world * plane.normal.view(1, 1, 3), dim=-1)
        valid = torch.abs(denom) > 1e-7
        t = -(torch.sum(cam_origin * plane.normal) + plane.d) / torch.where(
            valid, denom, torch.ones_like(denom)
        )
        valid = valid & (t > 0.0)

        points = cam_origin.view(1, 1, 3) + dirs_world * t.unsqueeze(-1)
        vec = points - plane.origin.view(1, 1, 3)
        u = torch.sum(vec * plane.e1.view(1, 1, 3), dim=-1)
        v = torch.sum(vec * plane.e2.view(1, 1, 3), dim=-1)
        atlas_x = (u - u_min) / du * x_den
        atlas_y = (v - v_min) / dv * y_den
        inside = valid & (atlas_x >= 0.0) & (atlas_x <= x_den) & (atlas_y >= 0.0) & (atlas_y <= y_den)

        grid_x = atlas_x / x_den * 2.0 - 1.0
        grid_y = atlas_y / y_den * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        sampled_texture = F.grid_sample(
            texture,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        sampled_mask = F.grid_sample(
            atlas_mask,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(0)
        sampled_mask_raw = F.grid_sample(
            atlas_mask_raw,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(0)
        mask_bool = (sampled_mask > 0.5) & inside
        raw_mask_bool = (sampled_mask_raw > 0.5) & inside

        rgb_chunk = (sampled_texture.permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        raw_mask_chunk = (raw_mask_bool.to(torch.uint8) * 255)
        mask_chunk = (mask_bool.to(torch.uint8) * 255)
        rgb_np = rgb_chunk.detach().cpu().numpy()
        raw_mask_np = raw_mask_chunk.detach().cpu().numpy()
        mask_np = mask_chunk.detach().cpu().numpy()
        rgb_np[mask_np == 0] = 0

        projected_rgb[row_start:row_end] = rgb_np
        projected_mask_raw[row_start:row_end] = raw_mask_np
        projected_mask[row_start:row_end] = mask_np

    return projected_rgb, projected_mask_raw, projected_mask


def overlay_projection_on_removal(
    removal_path: Path,
    projected_rgb: np.ndarray,
    projected_mask: np.ndarray,
    overlay_alpha: float,
) -> np.ndarray:
    with Image.open(removal_path) as image:
        removal = image.convert("RGB")
    if removal.size != (projected_rgb.shape[1], projected_rgb.shape[0]):
        projected_image = Image.fromarray(projected_rgb, mode="RGB").resize(removal.size, Image.Resampling.BILINEAR)
        projected_mask_image = Image.fromarray(projected_mask, mode="L").resize(removal.size, Image.Resampling.NEAREST)
        projected_rgb = np.asarray(projected_image, dtype=np.uint8)
        projected_mask = np.asarray(projected_mask_image, dtype=np.uint8)

    base = np.asarray(removal, dtype=np.float32)
    proj = projected_rgb.astype(np.float32)
    mask = projected_mask > 0
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    out = base.copy()
    if alpha >= 1.0:
        out[mask] = proj[mask]
    elif alpha > 0.0:
        out[mask] = base[mask] * (1.0 - alpha) + proj[mask] * alpha
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_arg}, but CUDA is not available.")
    return device


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    atlas_dir = resolve_child(args.model_path, args.desk_atlas_dir)
    debug_dir = resolve_child(args.model_path, args.debug_dir)
    projected_texture_dir = debug_dir / "projected_texture"
    projected_mask_raw_dir = debug_dir / "projected_mask_before_dilation"
    projected_mask_dir = debug_dir / "projected_mask"
    overlay_dir = debug_dir / "overlay_removal"
    debug_dir.mkdir(parents=True, exist_ok=True)
    projected_texture_dir.mkdir(parents=True, exist_ok=True)
    projected_mask_raw_dir.mkdir(parents=True, exist_ok=True)
    projected_mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    state = load_desk_atlas_state(atlas_dir / "desk_atlas_state.pt", device=device)
    texture = load_rgb_tensor(atlas_dir / args.texture_name, device=device)
    _, _, atlas_h, atlas_w = texture.shape
    raw_mask_image = load_binary_mask_image(atlas_dir / args.support_mask_name, 0)
    atlas_mask_raw = mask_image_to_tensor(raw_mask_image, (atlas_h, atlas_w), device=device)
    mask_image = load_binary_mask_image(
        atlas_dir / args.support_mask_name,
        args.mask_dilation,
    )
    atlas_mask = mask_image_to_tensor(mask_image, (atlas_h, atlas_w), device=device)
    save_atlas_debug(
        debug_dir,
        texture,
        atlas_mask_raw,
        atlas_mask,
        background_transparent=args.background_transparent,
    )

    cameras = load_cameras(args)
    removal_root = Path(args.removal_view_path)
    removal_index = build_removal_index(removal_root, recursive=args.recursive_removal_search)
    matched_cameras: List[Tuple[ProjectionCamera, Path]] = []
    skipped_cameras = []
    for camera in cameras:
        removal_path = resolve_removal_path(removal_root, removal_index, camera)
        if removal_path is None:
            skipped_cameras.append(camera.image_name)
            continue
        matched_cameras.append((camera, removal_path))
    if not matched_cameras:
        raise RuntimeError(
            f"No removal-view images under {removal_root} matched the "
            f"{len(cameras)} loaded cameras."
        )

    if skipped_cameras:
        print(
            f"[Info] Skipping {len(skipped_cameras)} cameras without matching "
            "removal views."
        )

    written_overlays = 0
    for index, (camera, removal_path) in enumerate(matched_cameras, start=1):
        print(f"[{index}/{len(matched_cameras)}] Projecting {camera.image_name}")
        projected_rgb, projected_mask_raw, projected_mask = project_atlas_to_camera(
            camera=camera,
            state=state,
            texture=texture,
            atlas_mask_raw=atlas_mask_raw,
            atlas_mask=atlas_mask,
            device=device,
            row_chunk=args.row_chunk,
        )
        output_name = debug_name(camera)
        if args.background_transparent:
            save_rgba(projected_texture_dir / output_name, projected_rgb, projected_mask)
        else:
            save_rgb(projected_texture_dir / output_name, projected_rgb)
        save_mask(
            projected_mask_raw_dir / output_name,
            projected_mask_raw,
            background_transparent=args.background_transparent,
        )
        save_mask(
            projected_mask_dir / output_name,
            projected_mask,
            background_transparent=args.background_transparent,
        )

        overlay = overlay_projection_on_removal(
            removal_path=removal_path,
            projected_rgb=projected_rgb,
            projected_mask=projected_mask,
            overlay_alpha=args.overlay_alpha,
        )
        save_rgb(overlay_dir / output_name, overlay)
        written_overlays += 1

    print(
        f"[Done] loaded_cameras={len(cameras)} projected_cameras={len(matched_cameras)} "
        f"skipped_missing_removal={len(skipped_cameras)} overlays={written_overlays} "
        f"debug_dir={debug_dir}"
    )


if __name__ == "__main__":
    main()
