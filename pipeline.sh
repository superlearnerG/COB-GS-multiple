#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 -s SOURCE_PATH -m MODEL_PATH [--support_object_ids IDS...] [--object_postprocess_skip_labels LABELS] [--removal_GT] [--removal_GT_ratio RATIO] [--id_in_background IDS...] [--only_desk+background] [--after_inpaint]"
  echo "       $0 --source_path SOURCE_PATH --model_path MODEL_PATH [--support_object_ids IDS...] [--object_postprocess_skip_labels LABELS] [--removal_GT] [--removal_GT_ratio RATIO] [--id_in_background IDS...] [--only_desk+background] [--after_inpaint]"
  echo "       IDS may be space-separated or comma-separated, for example: --support_object_ids 3 5 7"
  echo "       LABELS may be space-separated or comma-separated, for example: --object_postprocess_skip_labels 136 255"
}

SOURCE_PATH=""
MODEL_PATH=""
SUPPORT_OBJECT_IDS=()
OBJECT_POSTPROCESS_SKIP_LABELS=""
REMOVAL_GT=0
REMOVAL_GT_RATIO=""
ID_IN_BACKGROUND=()
ONLY_DESK_BACKGROUND=0
AFTER_INPAINT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source_path|--source-path)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      SOURCE_PATH="$2"
      shift 2
      ;;
    --source_path=*|--source-path=*)
      SOURCE_PATH="${1#*=}"
      shift
      ;;
    -m|--model_path|--model-path)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      MODEL_PATH="$2"
      shift 2
      ;;
    --model_path=*|--model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --support_object_ids|--support-object-ids)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      shift
      SUPPORT_OBJECT_IDS=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        SUPPORT_OBJECT_IDS+=("$1")
        shift
      done
      if [[ ${#SUPPORT_OBJECT_IDS[@]} -eq 0 ]]; then
        echo "Missing value for --support_object_ids" >&2
        usage >&2
        exit 1
      fi
      ;;
    --support_object_ids=*|--support-object-ids=*)
      SUPPORT_OBJECT_IDS=("${1#*=}")
      shift
      ;;
    --object_postprocess_skip_labels|--object-postprocess-skip-labels)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      shift
      skip_values=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        skip_values+=("$1")
        shift
      done
      if [[ ${#skip_values[@]} -eq 0 ]]; then
        echo "Missing value for --object_postprocess_skip_labels" >&2
        usage >&2
        exit 1
      fi
      OBJECT_POSTPROCESS_SKIP_LABELS="${skip_values[*]}"
      ;;
    --object_postprocess_skip_labels=*|--object-postprocess-skip-labels=*)
      OBJECT_POSTPROCESS_SKIP_LABELS="${1#*=}"
      shift
      ;;
    --removal_GT|--removal-gt)
      REMOVAL_GT=1
      shift
      ;;
    --removal_GT_ratio|--removal-gt-ratio)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      REMOVAL_GT_RATIO="$2"
      shift 2
      ;;
    --removal_GT_ratio=*|--removal-gt-ratio=*)
      REMOVAL_GT_RATIO="${1#*=}"
      shift
      ;;
    --id_in_background|--id-in-background)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      shift
      ID_IN_BACKGROUND=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        ID_IN_BACKGROUND+=("$1")
        shift
      done
      if [[ ${#ID_IN_BACKGROUND[@]} -eq 0 ]]; then
        echo "Missing value for --id_in_background" >&2
        usage >&2
        exit 1
      fi
      ;;
    --id_in_background=*|--id-in-background=*)
      ID_IN_BACKGROUND=("${1#*=}")
      shift
      ;;
    --only_desk+background|--only-desk-background|--only_desk_background)
      ONLY_DESK_BACKGROUND=1
      shift
      ;;
    --after_inpaint|--after-inpaint)
      AFTER_INPAINT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE_PATH" || -z "$MODEL_PATH" ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOURCE_PATH="${SOURCE_PATH%/}"
MODEL_PATH="${MODEL_PATH%/}"
MASK_ROOT="$SOURCE_PATH/object_mask"
START_CHECKPOINT="$MODEL_PATH/chkpnt30000.pth"
SEGMENTATION_CHECKPOINT="$MODEL_PATH/multi_object/final_multi_object.pth"
SUPPORT_OBJECT_ARGS=()
if [[ ${#SUPPORT_OBJECT_IDS[@]} -gt 0 ]]; then
  SUPPORT_OBJECT_ARGS=(--support_object_ids "${SUPPORT_OBJECT_IDS[@]}")
fi
OBJECT_POSTPROCESS_SKIP_ARGS=()
if [[ -n "$OBJECT_POSTPROCESS_SKIP_LABELS" ]]; then
  OBJECT_POSTPROCESS_SKIP_ARGS=(--object_postprocess_skip_labels "$OBJECT_POSTPROCESS_SKIP_LABELS")
fi
REMOVAL_GT_ARGS=()
if [[ "$REMOVAL_GT" -eq 1 ]]; then
  REMOVAL_GT_ARGS+=(--removal_GT)
fi
if [[ -n "$REMOVAL_GT_RATIO" ]]; then
  REMOVAL_GT_ARGS+=(--removal_GT_ratio "$REMOVAL_GT_RATIO")
fi
RENDER_ARGS=()
if [[ ${#ID_IN_BACKGROUND[@]} -gt 0 ]]; then
  RENDER_ARGS+=(--id_in_background "${ID_IN_BACKGROUND[@]}")
fi
if [[ "$ONLY_DESK_BACKGROUND" -eq 1 ]]; then
  RENDER_ARGS+=(--only_desk+background)
fi

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
#   -s "$SOURCE_PATH" \
#   -m "$MODEL_PATH" \
#   --eval

    # --use_depth_loss \

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
#   -s "$SOURCE_PATH" \
#   -m "$MODEL_PATH" \
#   --start_checkpoint "$START_CHECKPOINT" \
#   --include_mask \
#   --mask_mode multi_label \
#   --mask_root "$MASK_ROOT" \
#   --object_order label_asc \
#   --eval 

#   --object_debug_views 00034,00060,00062 \
#   --enable_object_postprocess \
#   "${OBJECT_POSTPROCESS_SKIP_ARGS[@]}" \
#   --object_postprocess_dilation_voxels 1 \
#   --object_postprocess_mask_thresh 0.0 \
#   --object_postprocess_voxel_scale 2 \
#   --finetune_mask

# TODO: add render to render results before inpaint

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
#   -s "$SOURCE_PATH" \
#   -m "$MODEL_PATH" \
#   --iteration 30000 \
#   --include_mask \
#   --mask_mode multi_label \
#   --segmentation_checkpoint "$SEGMENTATION_CHECKPOINT" \
#   --desk_object_id 255 \
#   --id_in_background 17 34 51 68 \
#   --background_transparent \
#   --only_removal \
#   "${RENDER_ARGS[@]}" \
#   --eval \
  # --only_desk+background \

    # --render_depth \


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python export_desk_atlas.py \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --mask_mode multi_label \
  --mask_root "$MASK_ROOT" \
  --object_order label_asc \
  --desk_object_id 255 \
  "${SUPPORT_OBJECT_ARGS[@]}" \
  --segmentation_checkpoint "$SEGMENTATION_CHECKPOINT" \
  --desk_atlas_long_side 1254 \
  --desk_atlas_size_multiple 6 \
  --ccm_max_mask_samples 1000000 \
  --desk_pack_known_strong_quantile 0.0 \
  --background_transparent \
  --eval 

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint.py \
#   -s "$SOURCE_PATH" \
#   -m "$MODEL_PATH" \
#   --desk_object_id 255 \
#   --source_iteration 30000 \
#   --iterations 32000 \
#   --segmentation_checkpoint "$SEGMENTATION_CHECKPOINT" \
#   --desk_atlas_dir "desk_atlas" \
#   --completed_texture_path "desk_atlas/texture_completed.png" \
#   "${REMOVAL_GT_ARGS[@]}" \
#   --save_training_vis \
#   --save_training_vis_iteration 500 \
#   --desk_support_shrink_px "0.0" \
#   --eval \

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
#   -m "$MODEL_PATH" \
#   --include_mask \
#   --mask_mode multi_label \
#   "${RENDER_ARGS[@]}" \
#   --eval

# if [[ "$AFTER_INPAINT" -eq 1 ]]; then
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python metrics.py \
#     -m "$MODEL_PATH" \
#     -s "$SOURCE_PATH" \
#     --after_inpaint
# fi
