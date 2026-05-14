import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from desk_atlas import (
    DeskAtlasState,
    FootprintMaps,
    PlaneDefinition,
    _compute_atlas_hw,
    _compute_known_masks_from_confidence,
    _compute_uv_bbox,
    _expand_uv_bbox_to_square,
    _filter_uv_outliers_iqr,
    _morph_erode,
    _morph_dilate,
    _normalize_single_channel_map,
    _poisson_like_prefill,
    _rasterize_uv_mask,
    _save_mask,
    _serialize_plane_definition,
    _unique_sorted_positive_labels,
    _uv_to_pixel_coords,
    build_desk_atlas_observation_state,
    build_object_footprint_on_plane,
    compute_plane_basis,
    fit_support_plane_from_visible_table,
    offset_plane_along_normal,
    pack_desk_atlas_for_texture_diffusion,
    save_desk_atlas_artifacts,
)
from utils.projection_utils import project_mask_pixels_to_plane_samples, project_xyz_to_plane_uv


PLANE_MODES = ("P0", "P1", "P2")
FUSION_MODES = ("F0", "F1", "F2", "F3", "F4")

ABLATION_SPECS = {
    "P0": ("P0", "F0", "plane_ransac_pca"),
    "P1": ("P1", "F0", "plane_pca_only"),
    "P2": ("P2", "F0", "plane_ransac_only"),
    "F0": ("P0", "F0", "fusion_bilinear_angle"),
    "F1": ("P0", "F1", "fusion_bilinear_uniform"),
    "F2": ("P0", "F2", "fusion_nearest_angle"),
    "F3": ("P0", "F3", "fusion_nearest_uniform"),
    "F4": ("P0", "F4", "fusion_max_confidence"),
}


@dataclass(frozen=True)
class AblationConfig:
    ablation: str
    plane_mode: str
    fusion_mode: str
    name: str


@dataclass
class _SampleChunk:
    uv: torch.Tensor
    values: torch.Tensor
    weight: torch.Tensor


def resolve_ablation_config(ablation: str) -> AblationConfig:
    key = str(ablation).upper()
    if key not in ABLATION_SPECS:
        raise ValueError(f"Unknown ablation '{ablation}'. Valid choices: {sorted(ABLATION_SPECS)}")
    plane_mode, fusion_mode, name = ABLATION_SPECS[key]
    return AblationConfig(ablation=key, plane_mode=plane_mode, fusion_mode=fusion_mode, name=name)


def default_output_subdir(config: AblationConfig) -> str:
    return os.path.join("desk_atlas_ablation", f"{config.ablation}_{config.name}")


def _fit_plane_pca_only(table_points: torch.Tensor) -> PlaneDefinition:
    if table_points.shape[0] < 3:
        raise RuntimeError("Not enough points to fit PCA-only support plane")

    origin = table_points.mean(dim=0)
    centered = table_points - origin[None]
    cov = (centered.t() @ centered) / max(1, centered.shape[0])
    _, eigvecs = torch.linalg.eigh(cov)
    normal = F.normalize(eigvecs[:, 0], dim=0)
    if normal[2] < 0:
        normal = -normal
    d = -torch.dot(normal, origin)
    e1, e2 = compute_plane_basis(normal)
    return PlaneDefinition(normal=normal, d=d, origin=origin, e1=e1, e2=e2)


def _fit_plane_ransac_only(
    table_points: torch.Tensor,
    ransac_iters: int,
    inlier_thresh: float,
) -> PlaneDefinition:
    if table_points.shape[0] < 3:
        raise RuntimeError("Not enough points to fit RANSAC-only support plane")

    n_points = int(table_points.shape[0])
    best_normal = None
    best_d = None
    best_inlier_mask = None
    best_count = -1

    for _ in range(max(32, int(ransac_iters))):
        sample_idx = torch.randint(0, n_points, (3,), device=table_points.device)
        p0, p1, p2 = table_points[sample_idx]
        normal = torch.cross(p1 - p0, p2 - p0, dim=0)
        norm_val = torch.norm(normal)
        if norm_val < 1e-7:
            continue
        normal = normal / norm_val
        if normal[2] < 0:
            normal = -normal
        d = -torch.dot(normal, p0)
        dist = torch.abs(table_points @ normal + d)
        inliers = dist < float(inlier_thresh)
        count = int(inliers.sum().item())
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d
            best_inlier_mask = inliers

    if best_normal is None or best_d is None:
        return _fit_plane_pca_only(table_points)

    if best_inlier_mask is not None and int(best_inlier_mask.sum().item()) > 0:
        mean_point = table_points[best_inlier_mask].mean(dim=0)
    else:
        mean_point = table_points.mean(dim=0)
    # Keep the RANSAC hypothesis normal/d, but choose a stable origin on that plane.
    origin = mean_point - (torch.dot(mean_point, best_normal) + best_d) * best_normal
    e1, e2 = compute_plane_basis(best_normal)
    return PlaneDefinition(normal=best_normal, d=best_d, origin=origin, e1=e1, e2=e2)


def fit_ablation_support_plane(
    table_points: torch.Tensor,
    plane_mode: str,
    ransac_iters: int,
    inlier_thresh: float,
) -> PlaneDefinition:
    mode = str(plane_mode).upper()
    if mode == "P0":
        return fit_support_plane_from_visible_table(
            table_points,
            ransac_iters=ransac_iters,
            inlier_thresh=inlier_thresh,
        )
    if mode == "P1":
        return _fit_plane_pca_only(table_points)
    if mode == "P2":
        return _fit_plane_ransac_only(
            table_points,
            ransac_iters=ransac_iters,
            inlier_thresh=inlier_thresh,
        )
    raise ValueError(f"Unsupported plane ablation mode '{plane_mode}'")


def _camera_source_key(camera) -> str:
    image_name = getattr(camera, "image_name", None)
    if image_name is not None:
        return str(image_name)
    return str(getattr(camera, "uid", id(camera)))


def _get_camera_source_map(
    camera,
    source_kind: str,
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Optional[torch.Tensor]:
    if source_maps_by_camera is not None:
        source_maps = source_maps_by_camera.get(_camera_source_key(camera), {})
        if source_kind in source_maps:
            return source_maps[source_kind]
    if source_kind == "rgb":
        return getattr(camera, "original_image", None)
    raise ValueError(f"Unsupported atlas source kind '{source_kind}'")


def _collect_ablation_visible_desk_sample_chunks(
    scene,
    mask_provider,
    plane: PlaneDefinition,
    desk_object_id: int,
    max_samples: int,
    fusion_mode: str,
    source_kind: str = "rgb",
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> List[_SampleChunk]:
    chunks: List[_SampleChunk] = []

    for cam in scene.getTrainCameras():
        source_image = _get_camera_source_map(cam, source_kind, source_maps_by_camera=source_maps_by_camera)
        if source_image is None:
            continue

        label_map = mask_provider.get_label_map(cam).squeeze(0).long().to(cam.camera_center.device)
        desk_mask = label_map == int(desk_object_id)
        uv, values, weight = project_mask_pixels_to_plane_samples(
            desk_mask,
            cam,
            plane,
            max_samples=max_samples,
            source_image=source_image,
        )
        if uv.shape[0] == 0:
            del uv, values, weight, label_map, desk_mask
            continue
        if str(fusion_mode).upper() in {"F1", "F3"}:
            weight = torch.ones_like(weight)
        chunks.append(
            _SampleChunk(
                uv=uv.detach().cpu(),
                values=values.detach().cpu(),
                weight=weight.detach().cpu(),
            )
        )
        del uv, values, weight, label_map, desk_mask

    return chunks


def _compute_uv_bbox_from_sample_chunks(
    chunks: Sequence[_SampleChunk],
    extra_uv: Optional[torch.Tensor] = None,
    pad_ratio: float = 0.05,
) -> Tuple[float, float, float, float]:
    u_min = float("inf")
    u_max = -float("inf")
    v_min = float("inf")
    v_max = -float("inf")

    def update_from_uv(uv: torch.Tensor) -> None:
        nonlocal u_min, u_max, v_min, v_max
        if uv is None or uv.shape[0] == 0:
            return
        uv_cpu = uv.detach().cpu()
        u_min = min(u_min, float(uv_cpu[:, 0].min().item()))
        u_max = max(u_max, float(uv_cpu[:, 0].max().item()))
        v_min = min(v_min, float(uv_cpu[:, 1].min().item()))
        v_max = max(v_max, float(uv_cpu[:, 1].max().item()))

    for chunk in chunks:
        update_from_uv(chunk.uv)
    update_from_uv(extra_uv)

    if not math.isfinite(u_min) or not math.isfinite(v_min):
        raise RuntimeError("Cannot compute uv bbox from empty samples")

    du = max(u_max - u_min, 1e-6)
    dv = max(v_max - v_min, 1e-6)
    pad_u = float(pad_ratio) * du
    pad_v = float(pad_ratio) * dv
    return u_min - pad_u, u_max + pad_u, v_min - pad_v, v_max + pad_v


def _accumulate_uv_mask_heat(
    heat: torch.Tensor,
    uv: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> None:
    if uv.shape[0] == 0:
        return
    x, y = _uv_to_pixel_coords(uv, bbox, height, width)
    xi = x.round().long().clamp_(0, width - 1)
    yi = y.round().long().clamp_(0, height - 1)
    heat.index_put_((yi, xi), torch.ones_like(xi, dtype=torch.float32), accumulate=True)


def _rasterize_sample_chunks_uv_mask(
    chunks: Sequence[_SampleChunk],
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    kernel: int,
    device: torch.device,
) -> torch.Tensor:
    heat = torch.zeros((height, width), dtype=torch.float32, device=device)
    for chunk in chunks:
        uv = chunk.uv.to(device=device)
        _accumulate_uv_mask_heat(heat, uv, bbox, height, width)
        del uv
    mask = heat > 0
    close_kernel = max(3, int(kernel) | 1)
    return _morph_erode(_morph_dilate(mask, close_kernel), close_kernel)


def _accumulate_samples(
    flat_idx: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    channel_count: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value_sum = torch.zeros((channel_count, height * width), dtype=values.dtype, device=device)
    weight_sum = torch.zeros((height * width,), dtype=values.dtype, device=device)
    weight_sum.index_put_((flat_idx,), weights, accumulate=True)
    for channel in range(channel_count):
        value_sum[channel].index_put_((flat_idx,), values[:, channel] * weights, accumulate=True)
    return value_sum.view(channel_count, height, width), weight_sum.view(height, width)


def _accumulate_flat_samples_inplace(
    value_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    flat_idx: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    value_sum_flat = value_sum.view(value_sum.shape[0], -1)
    weight_sum_flat = weight_sum.view(-1)
    weight_sum_flat.index_put_((flat_idx,), weights, accumulate=True)
    for channel in range(int(value_sum.shape[0])):
        value_sum_flat[channel].index_put_((flat_idx,), values[:, channel] * weights, accumulate=True)


def _accumulate_max_confidence_flat_inplace(
    value_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    best_score: torch.Tensor,
    flat_idx: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    if flat_idx.numel() == 0:
        return

    pixel_count = int(best_score.numel())
    batch_best = torch.full((pixel_count,), -float("inf"), dtype=scores.dtype, device=scores.device)
    if hasattr(batch_best, "scatter_reduce_"):
        batch_best.scatter_reduce_(0, flat_idx, scores, reduce="amax", include_self=True)
    else:
        for idx, score in zip(flat_idx.tolist(), scores.tolist()):
            if score > float(batch_best[idx].item()):
                batch_best[idx] = score

    reset_idx = (batch_best > best_score).nonzero(as_tuple=False).squeeze(1)
    if reset_idx.numel() > 0:
        best_score[reset_idx] = batch_best[reset_idx]
        weight_sum.view(-1)[reset_idx] = 0
        value_sum.view(value_sum.shape[0], -1)[:, reset_idx] = 0

    keep = scores >= best_score[flat_idx]
    if keep.any():
        _accumulate_flat_samples_inplace(
            value_sum=value_sum,
            weight_sum=weight_sum,
            flat_idx=flat_idx[keep],
            values=values[keep],
            weights=scores[keep],
        )


def _bilinear_splat_indices_and_weights(
    uv: torch.Tensor,
    weight: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = _uv_to_pixel_coords(uv, bbox, height, width)
    x0 = torch.floor(x).long().clamp_(0, width - 1)
    y0 = torch.floor(y).long().clamp_(0, height - 1)
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)

    wx1 = x - x0.float()
    wy1 = y - y0.float()
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    flat_idx = torch.cat(
        (
            y0 * width + x0,
            y0 * width + x1,
            y1 * width + x0,
            y1 * width + x1,
        ),
        dim=0,
    )
    splat_weight = torch.cat(
        (
            weight * wx0 * wy0,
            weight * wx1 * wy0,
            weight * wx0 * wy1,
            weight * wx1 * wy1,
        ),
        dim=0,
    )
    return flat_idx, splat_weight


def _max_confidence_samples(
    flat_idx: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    channel_count: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pixel_count = height * width
    best_score = torch.full((pixel_count,), -float("inf"), dtype=scores.dtype, device=device)
    if hasattr(best_score, "scatter_reduce_"):
        best_score.scatter_reduce_(0, flat_idx, scores, reduce="amax", include_self=True)
    else:
        for idx, score in zip(flat_idx.tolist(), scores.tolist()):
            if score > float(best_score[idx].item()):
                best_score[idx] = score

    keep = scores >= best_score[flat_idx]
    kept_idx = flat_idx[keep]
    kept_scores = scores[keep]
    kept_values = values[keep]
    value_sum, weight_sum = _accumulate_samples(
        kept_idx,
        kept_values,
        kept_scores,
        channel_count,
        height,
        width,
        device,
    )
    weight_sum = torch.where(torch.isfinite(weight_sum), weight_sum, torch.zeros_like(weight_sum))
    return value_sum, weight_sum


def _nearest_splat_values_to_atlas(
    uv: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    channel_count = int(values.shape[1]) if values.ndim == 2 else 3
    value_sum = torch.zeros((channel_count, height, width), dtype=values.dtype, device=device)
    weight_sum = torch.zeros((height, width), dtype=values.dtype, device=device)
    if uv.shape[0] == 0:
        return value_sum, weight_sum

    x, y = _uv_to_pixel_coords(uv, bbox, height, width)
    xi = x.round().long().clamp_(0, width - 1)
    yi = y.round().long().clamp_(0, height - 1)
    flat_idx = yi * width + xi
    return _accumulate_samples(flat_idx, values, weight, channel_count, height, width, device)


def _bilinear_or_max_splat_values_to_atlas(
    uv: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
    max_confidence: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    channel_count = int(values.shape[1]) if values.ndim == 2 else 3
    value_sum = torch.zeros((channel_count, height, width), dtype=values.dtype, device=device)
    weight_sum = torch.zeros((height, width), dtype=values.dtype, device=device)
    if uv.shape[0] == 0:
        return value_sum, weight_sum

    x, y = _uv_to_pixel_coords(uv, bbox, height, width)
    x0 = torch.floor(x).long().clamp_(0, width - 1)
    y0 = torch.floor(y).long().clamp_(0, height - 1)
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)

    wx1 = x - x0.float()
    wy1 = y - y0.float()
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    corners = (
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    )
    flat_parts = []
    value_parts = []
    weight_parts = []
    for xi, yi, corner_weight in corners:
        cur_weight = weight * corner_weight
        flat_parts.append(yi * width + xi)
        value_parts.append(values)
        weight_parts.append(cur_weight)

    flat_idx = torch.cat(flat_parts, dim=0)
    splat_values = torch.cat(value_parts, dim=0)
    splat_weight = torch.cat(weight_parts, dim=0)
    if max_confidence:
        return _max_confidence_samples(
            flat_idx,
            splat_values,
            splat_weight,
            channel_count,
            height,
            width,
            device,
        )
    return _accumulate_samples(flat_idx, splat_values, splat_weight, channel_count, height, width, device)


def splat_ablation_values_to_atlas(
    uv: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
    fusion_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = str(fusion_mode).upper()
    if mode in {"F0", "F1"}:
        return _bilinear_or_max_splat_values_to_atlas(
            uv,
            values,
            weight,
            bbox,
            height,
            width,
            device,
            max_confidence=False,
        )
    if mode in {"F2", "F3"}:
        return _nearest_splat_values_to_atlas(uv, values, weight, bbox, height, width, device)
    if mode == "F4":
        return _bilinear_or_max_splat_values_to_atlas(
            uv,
            values,
            weight,
            bbox,
            height,
            width,
            device,
            max_confidence=True,
        )
    raise ValueError(f"Unsupported fusion ablation mode '{fusion_mode}'")


def _accumulate_ablation_values_to_atlas_inplace(
    value_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    best_score: Optional[torch.Tensor],
    uv: torch.Tensor,
    values: torch.Tensor,
    weight: torch.Tensor,
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    fusion_mode: str,
) -> None:
    if uv.shape[0] == 0:
        return
    if values.ndim == 1:
        values = values[:, None]

    mode = str(fusion_mode).upper()
    if mode in {"F0", "F1", "F4"}:
        flat_idx, splat_weight = _bilinear_splat_indices_and_weights(
            uv,
            weight,
            bbox,
            height,
            width,
        )
        splat_values = values.repeat(4, 1)
        if mode == "F4":
            if best_score is None:
                raise ValueError("F4 accumulation requires best_score.")
            _accumulate_max_confidence_flat_inplace(
                value_sum=value_sum,
                weight_sum=weight_sum,
                best_score=best_score,
                flat_idx=flat_idx,
                values=splat_values,
                scores=splat_weight,
            )
        else:
            _accumulate_flat_samples_inplace(
                value_sum=value_sum,
                weight_sum=weight_sum,
                flat_idx=flat_idx,
                values=splat_values,
                weights=splat_weight,
            )
        return

    if mode in {"F2", "F3"}:
        x, y = _uv_to_pixel_coords(uv, bbox, height, width)
        xi = x.round().long().clamp_(0, width - 1)
        yi = y.round().long().clamp_(0, height - 1)
        flat_idx = yi * width + xi
        _accumulate_flat_samples_inplace(
            value_sum=value_sum,
            weight_sum=weight_sum,
            flat_idx=flat_idx,
            values=values,
            weights=weight,
        )
        return

    raise ValueError(f"Unsupported fusion ablation mode '{fusion_mode}'")


def _splat_sample_chunks_to_atlas(
    chunks: Sequence[_SampleChunk],
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
    fusion_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(chunks) == 0:
        raise RuntimeError("No visible desk pixels available for atlas construction")

    first_values = chunks[0].values
    channel_count = int(first_values.shape[1]) if first_values.ndim == 2 else 3
    dtype = first_values.dtype if first_values.is_floating_point() else torch.float32
    value_sum = torch.zeros((channel_count, height, width), dtype=dtype, device=device)
    weight_sum = torch.zeros((height, width), dtype=dtype, device=device)
    best_score = None
    if str(fusion_mode).upper() == "F4":
        best_score = torch.full((height * width,), -float("inf"), dtype=dtype, device=device)

    for chunk in chunks:
        uv = chunk.uv.to(device=device)
        values = chunk.values.to(device=device, dtype=dtype)
        weight = chunk.weight.to(device=device, dtype=dtype)
        _accumulate_ablation_values_to_atlas_inplace(
            value_sum=value_sum,
            weight_sum=weight_sum,
            best_score=best_score,
            uv=uv,
            values=values,
            weight=weight,
            bbox=bbox,
            height=height,
            width=width,
            fusion_mode=fusion_mode,
        )
        del uv, values, weight

    return value_sum, weight_sum


def build_desk_atlas_ablation_state(
    scene,
    gaussians,
    mask_provider,
    opt,
    desk_object_id: int,
    support_object_ids: Sequence[int],
    iteration: int,
    config: AblationConfig,
    source_maps_by_camera: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> DeskAtlasState:
    desk_mask = gaussians.get_object_id == int(desk_object_id)
    if desk_mask.sum() < 3:
        raise RuntimeError(
            f"Not enough desk Gaussians to fit plane for desk_object_id={int(desk_object_id)}; "
            f"found {int(desk_mask.sum().item())}."
        )

    plane = fit_ablation_support_plane(
        gaussians._xyz.detach()[desk_mask],
        plane_mode=config.plane_mode,
        ransac_iters=opt.ccm_plane_ransac_iters,
        inlier_thresh=opt.ccm_plane_ransac_thresh,
    )
    plane_down_offset = float(getattr(opt, "desk_plane_down_offset", 0.0))
    if plane_down_offset != 0.0:
        plane = offset_plane_along_normal(plane, -plane_down_offset)

    visible_chunks = _collect_ablation_visible_desk_sample_chunks(
        scene=scene,
        mask_provider=mask_provider,
        plane=plane,
        desk_object_id=int(desk_object_id),
        max_samples=opt.ccm_max_mask_samples,
        fusion_mode=config.fusion_mode,
        source_maps_by_camera=source_maps_by_camera,
    )
    if len(visible_chunks) == 0:
        raise RuntimeError("No visible desk pixels available for atlas construction")

    footprint = build_object_footprint_on_plane(
        gaussians=gaussians,
        support_object_ids=support_object_ids,
        plane=plane,
        bottom_quantile=opt.footprint_bottom_quantile,
        min_opacity=opt.footprint_min_opacity,
        uv_outlier_iqr=opt.footprint_uv_outlier_iqr,
    )

    raw_uv_bbox = _compute_uv_bbox_from_sample_chunks(
        visible_chunks,
        extra_uv=footprint.uv_points if footprint.uv_points.shape[0] > 0 else None,
    )
    uv_bbox = _expand_uv_bbox_to_square(raw_uv_bbox)
    atlas_h, atlas_w = _compute_atlas_hw(
        uv_bbox,
        opt.desk_atlas_long_side,
        opt.desk_atlas_size_multiple,
    )

    support_visible = _rasterize_sample_chunks_uv_mask(
        visible_chunks,
        bbox=uv_bbox,
        height=atlas_h,
        width=atlas_w,
        kernel=opt.ccm_contact_kernel,
        device=gaussians.get_xyz.device,
    )
    support_footprint = _rasterize_uv_mask(
        footprint.uv_points,
        bbox=uv_bbox,
        height=atlas_h,
        width=atlas_w,
        kernel=opt.ccm_contact_kernel,
        device=gaussians.get_xyz.device,
    )
    support_mask = torch.logical_or(support_visible, support_footprint)

    rgb_sum, confidence = _splat_sample_chunks_to_atlas(
        visible_chunks,
        bbox=uv_bbox,
        height=atlas_h,
        width=atlas_w,
        device=gaussians.get_xyz.device,
        fusion_mode=config.fusion_mode,
    )
    rgb_observed = rgb_sum / confidence.clamp_min(1e-6).unsqueeze(0)
    observed_mask = confidence > 0
    rgb_observed = rgb_observed * observed_mask.unsqueeze(0).float()
    hole_mask = torch.logical_and(support_mask, ~observed_mask)

    state = DeskAtlasState(
        plane=plane,
        uv_bbox=uv_bbox,
        atlas_hw=(atlas_h, atlas_w),
        support_mask=support_mask,
        support_visible_mask=support_visible,
        support_footprint_mask=support_footprint,
        observed_mask=observed_mask,
        hole_mask=hole_mask,
        confidence=confidence,
        rgb_observed=rgb_observed,
        build_iteration=int(iteration),
    )
    print(
        "[DeskAtlasAblation] "
        f"ablation={config.ablation} "
        f"plane_mode={config.plane_mode} "
        f"fusion_mode={config.fusion_mode} "
        f"build_iteration={int(iteration)} "
        f"desk_object_id={int(desk_object_id)} "
        f"support_object_ids={list(int(v) for v in support_object_ids)} "
        f"atlas_hw=({atlas_h}, {atlas_w}) "
        f"raw_uv_bbox={tuple(round(float(v), 6) for v in raw_uv_bbox)} "
        f"square_uv_bbox={tuple(round(float(v), 6) for v in uv_bbox)} "
        f"desk_plane_down_offset={plane_down_offset:.6g} "
        f"support_pixels={int(support_mask.sum().item())} "
        f"observed_pixels={int(observed_mask.sum().item())}"
    )
    return state


def export_desk_atlas_ablation_modalities(
    model_path: str,
    desk_atlas_state: DeskAtlasState,
    opt,
    output_subdir: str,
    config: AblationConfig,
) -> Dict[str, Dict[str, Any]]:
    diffusion_pack = pack_desk_atlas_for_texture_diffusion(
        desk_atlas_state,
        known_strong_quantile=float(opt.desk_pack_known_strong_quantile),
        hole_observed_dilate_kernel=int(getattr(opt, "desk_pack_hole_observed_dilate_kernel", 1)),
        source_kind="rgb",
    )
    outputs = {"rgb": {"state": desk_atlas_state, "pack": diffusion_pack}}
    save_desk_atlas_artifacts(model_path, desk_atlas_state, outputs, output_subdir=output_subdir)
    _write_ablation_metadata(model_path, output_subdir, config)
    return outputs


def _write_ablation_metadata(model_path: str, output_subdir: str, config: AblationConfig) -> None:
    atlas_dir = output_subdir if os.path.isabs(output_subdir) else os.path.join(model_path, output_subdir)
    payload = {
        "ablation": config.ablation,
        "plane_mode": config.plane_mode,
        "fusion_mode": config.fusion_mode,
        "name": config.name,
        "plane_modes": {
            "P0": "RANSAC inlier selection followed by PCA refinement",
            "P1": "PCA-only over all desk Gaussians",
            "P2": "RANSAC best hypothesis without PCA refinement",
        },
        "fusion_modes": {
            "F0": "bilinear splatting with angle weight abs(ray_dot_plane_normal)",
            "F1": "bilinear splatting with uniform weight",
            "F2": "nearest-pixel splatting with angle weight abs(ray_dot_plane_normal)",
            "F3": "nearest-pixel splatting with uniform weight",
            "F4": "bilinear support with max-confidence sample selection per atlas pixel",
        },
    }
    with open(os.path.join(atlas_dir, "ablation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
