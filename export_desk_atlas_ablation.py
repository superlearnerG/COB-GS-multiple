import os
import random
from argparse import ArgumentParser, Namespace

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from desk_atlas_ablation import (
    ABLATION_SPECS,
    build_desk_atlas_ablation_state,
    default_output_subdir,
    export_desk_atlas_ablation_modalities,
    resolve_ablation_config,
)
from export_desk_atlas import (
    _ensure_dataset_defaults,
    _resolve_checkpoint_path,
    load_segmentation_checkpoint,
    parse_object_id_list,
    parse_single_desk_object_id,
)


def _add_desk_atlas_ablation_args(parser: ArgumentParser) -> None:
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--segmentation_checkpoint", default=None, type=str)
    parser.add_argument("--desk_object_id", default=None)
    parser.add_argument("--support_object_ids", nargs="+", default=None)
    parser.add_argument(
        "--ablation",
        default="P0",
        choices=sorted(ABLATION_SPECS),
        help=(
            "Fixed desk-atlas ablation choice. "
            "P0/P1/P2 compare plane fitting with F0 fixed; "
            "F0/F1/F2/F3/F4 compare fusion with P0 fixed."
        ),
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Output atlas directory. Defaults to desk_atlas_ablation/<ablation>_<name>.",
    )
    parser.add_argument("--desk_atlas_long_side", type=int, default=1024)
    parser.add_argument("--desk_atlas_size_multiple", type=int, default=8)
    parser.add_argument("--footprint_bottom_quantile", type=float, default=1.0)
    parser.add_argument("--footprint_min_opacity", type=float, default=0.05)
    parser.add_argument("--footprint_uv_outlier_iqr", type=float, default=1.5)
    parser.add_argument("--ccm_contact_kernel", type=int, default=15)
    parser.add_argument("--ccm_max_mask_samples", type=int, default=80000)
    parser.add_argument("--ccm_plane_ransac_iters", type=int, default=256)
    parser.add_argument("--ccm_plane_ransac_thresh", type=float, default=0.01)
    parser.add_argument("--desk_plane_down_offset", type=float, default=0.0)
    parser.add_argument("--desk_pack_known_strong_quantile", type=float, default=0.0)
    parser.add_argument("--desk_pack_hole_observed_dilate_kernel", type=int, default=5)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--list_ablations",
        action="store_true",
        help="Print the fixed ablation mapping and exit.",
    )


def _extract_desk_atlas_ablation_options(args) -> Namespace:
    return Namespace(
        desk_atlas_long_side=int(args.desk_atlas_long_side),
        desk_atlas_size_multiple=int(args.desk_atlas_size_multiple),
        footprint_bottom_quantile=float(args.footprint_bottom_quantile),
        footprint_min_opacity=float(args.footprint_min_opacity),
        footprint_uv_outlier_iqr=float(args.footprint_uv_outlier_iqr),
        ccm_contact_kernel=int(args.ccm_contact_kernel),
        ccm_max_mask_samples=int(args.ccm_max_mask_samples),
        ccm_plane_ransac_iters=int(args.ccm_plane_ransac_iters),
        ccm_plane_ransac_thresh=float(args.ccm_plane_ransac_thresh),
        desk_plane_down_offset=float(args.desk_plane_down_offset),
        desk_pack_known_strong_quantile=float(args.desk_pack_known_strong_quantile),
        desk_pack_hole_observed_dilate_kernel=int(args.desk_pack_hole_observed_dilate_kernel),
    )


def print_ablation_mapping() -> None:
    for key in sorted(ABLATION_SPECS):
        config = resolve_ablation_config(key)
        print(
            f"{key}: plane_mode={config.plane_mode} "
            f"fusion_mode={config.fusion_mode} "
            f"name={config.name}"
        )


def export_desk_atlas_ablation(dataset, pipe, args) -> None:
    del pipe

    from desk_atlas import available_object_ids, infer_support_object_ids, validate_desk_object_id
    from scene import GaussianModel, Scene
    from utils.mask_provider import MultiLabelMaskProvider

    config = resolve_ablation_config(getattr(args, "ablation", "P0"))
    desk_object_id = parse_single_desk_object_id(getattr(args, "desk_object_id", None))
    output_subdir = getattr(args, "output_subdir", None) or default_output_subdir(config)
    opt = _extract_desk_atlas_ablation_options(args)
    _ensure_dataset_defaults(dataset)

    if not os.path.isdir(dataset.model_path):
        raise FileNotFoundError(f"Model path does not exist: {dataset.model_path}")
    if not os.path.isdir(dataset.mask_root):
        raise FileNotFoundError(f"Mask root does not exist: {dataset.mask_root}")

    gaussians = GaussianModel(dataset.sh_degree)
    load_iteration = getattr(args, "iteration", -1)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

    checkpoint_path = _resolve_checkpoint_path(dataset.model_path, getattr(args, "segmentation_checkpoint", None))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Segmentation checkpoint not found: {checkpoint_path}. "
            "Run multi-object segmentation first, or pass --segmentation_checkpoint explicitly."
        )

    checkpoint_marker = load_segmentation_checkpoint(scene.gaussians, checkpoint_path)
    desk_object_id = validate_desk_object_id(scene.gaussians, desk_object_id)
    explicit_support_ids = parse_object_id_list(getattr(args, "support_object_ids", None))
    support_object_ids = infer_support_object_ids(
        scene.gaussians,
        desk_object_id=desk_object_id,
        explicit_labels=explicit_support_ids if explicit_support_ids else None,
    )

    mask_provider = MultiLabelMaskProvider(
        dataset.mask_root,
        scene.getTrainCameras(),
        target_labels="",
        object_order=dataset.object_order,
    )
    visible_mask_labels = set(mask_provider.get_object_labels())
    if desk_object_id not in visible_mask_labels:
        print(
            f"[DeskAtlasAblation] Warning: desk_object_id={desk_object_id} is not present "
            f"in 2D masks under '{dataset.mask_root}'."
        )

    build_iteration = int(scene.loaded_iter or checkpoint_marker or 0)
    with torch.no_grad():
        desk_atlas_state = build_desk_atlas_ablation_state(
            scene=scene,
            gaussians=scene.gaussians,
            mask_provider=mask_provider,
            opt=opt,
            desk_object_id=desk_object_id,
            support_object_ids=support_object_ids,
            iteration=build_iteration,
            config=config,
        )
        export_desk_atlas_ablation_modalities(
            model_path=dataset.model_path,
            desk_atlas_state=desk_atlas_state,
            opt=opt,
            output_subdir=output_subdir,
            config=config,
        )

    output_dir = output_subdir if os.path.isabs(output_subdir) else os.path.join(dataset.model_path, output_subdir)
    print(
        "[DeskAtlasAblation] Export complete "
        f"ablation={config.ablation} "
        f"plane_mode={config.plane_mode} "
        f"fusion_mode={config.fusion_mode} "
        f"segmentation_checkpoint='{checkpoint_path}' "
        f"desk_object_id={desk_object_id} "
        f"support_object_ids={support_object_ids} "
        f"available_object_ids={available_object_ids(scene.gaussians)} "
        f"output_dir='{output_dir}'"
    )


def main() -> None:
    parser = ArgumentParser(description="Export fixed internal desk-atlas ablation artifacts from mod-COB-GS.")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    _add_desk_atlas_ablation_args(parser)
    args = get_combined_args(parser)

    if getattr(args, "list_ablations", False):
        print_ablation_mapping()
        return

    if not getattr(args, "source_path", None):
        parser.error("--source_path is required")
    if not getattr(args, "model_path", None):
        parser.error("--model_path is required")

    random_seed = int(getattr(args, "seed", 0) or 0)

    import numpy as np
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    export_desk_atlas_ablation(lp.extract(args), pp.extract(args), args)


if __name__ == "__main__":
    main()
