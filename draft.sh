#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS=(
  "office_desk"
  "dining_table"
  "bedroom"
)

ABLATION="P1_plane_pca_only"
DATA_ROOT="../data"
OUTPUT_ROOT="../output"
MODEL_SUBDIR="cobgs/5010000"

declare -A ID_IN_BACKGROUND_BY_DATASET=(
  ["office_desk"]="136"
  ["dining_table"]="17 34 51 68"
  ["bedroom"]=""
)

SOURCE_ITERATION=30000
TARGET_ITERATION=32000
DESK_OBJECT_ID=255
DESK_SUPPORT_SHRINK_PX=60

for idx in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$idx]}"
  CURRENT_SOURCE="$(realpath -m "$DATA_ROOT/$DATASET")"
  BASE_MODEL="$(realpath -m "$OUTPUT_ROOT/$DATASET/$MODEL_SUBDIR")"
  RUN_MODEL="$(realpath -m "$BASE_MODEL/desk_atlas_ablation/$ABLATION/inpaint_model")"
  ATLAS_DIR="$(dirname "$RUN_MODEL")"
  SOURCE_POINT_CLOUD="$(realpath -e "$BASE_MODEL/point_cloud/iteration_${SOURCE_ITERATION}")"
  SEGMENTATION_CHECKPOINT="$(realpath -e "$BASE_MODEL/multi_object/final_multi_object.pth")"
  ID_IN_BACKGROUND_RAW="${ID_IN_BACKGROUND_BY_DATASET[$DATASET]:-}"
  ID_IN_BACKGROUND_ARGS=()
  if [[ -n "$ID_IN_BACKGROUND_RAW" ]]; then
    read -r -a ID_IN_BACKGROUND_VALUES <<< "$ID_IN_BACKGROUND_RAW"
    ID_IN_BACKGROUND_ARGS=(--id_in_background "${ID_IN_BACKGROUND_VALUES[@]}")
  fi

  echo "============================================================"
  echo "[RUN $((idx + 1))/${#DATASETS[@]}] dataset=$DATASET ablation=$ABLATION"
  echo "SOURCE=$CURRENT_SOURCE"
  echo "BASE_MODEL=$BASE_MODEL"
  echo "ATLAS_DIR=$ATLAS_DIR"
  echo "RUN_MODEL=$RUN_MODEL"
  echo "ID_IN_BACKGROUND=${ID_IN_BACKGROUND_RAW:-<none>}"

  mkdir -p "$RUN_MODEL/point_cloud"
  ln -sfnT "$SOURCE_POINT_CLOUD" "$RUN_MODEL/point_cloud/iteration_${SOURCE_ITERATION}"
  cp "$BASE_MODEL/cfg_args" "$RUN_MODEL/cfg_args"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint.py \
    -s "$CURRENT_SOURCE" \
    -m "$RUN_MODEL" \
    --desk_object_id "$DESK_OBJECT_ID" \
    --source_iteration "$SOURCE_ITERATION" \
    --iterations "$TARGET_ITERATION" \
    --segmentation_checkpoint "$SEGMENTATION_CHECKPOINT" \
    --desk_atlas_dir "$ATLAS_DIR" \
    --completed_texture_path "$ATLAS_DIR/texture_completed.png" \
    --desk_support_shrink_px "$DESK_SUPPORT_SHRINK_PX" \
    --save_training_vis \
    --save_training_vis_iteration 500 \
    --eval

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
    -s "$CURRENT_SOURCE" \
    -m "$RUN_MODEL" \
    --iteration "$TARGET_ITERATION" \
    --include_mask \
    --mask_mode multi_label \
    --desk_object_id "$DESK_OBJECT_ID" \
    --segmentation_checkpoint "$RUN_MODEL/point_cloud/iteration_${TARGET_ITERATION}/point_cloud.pth" \
    "${ID_IN_BACKGROUND_ARGS[@]}" \
    --only_desk+background \
    --eval
done
