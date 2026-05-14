#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash pipeline_wrapper.sh
  bash pipeline_wrapper.sh --pair SOURCE_1 MODEL_1 [SUPPORT_IDS_1] [--pair SOURCE_2 MODEL_2 [SUPPORT_IDS_2] ...]
  bash pipeline_wrapper.sh [--removal_GT] [--removal_GT_ratio RATIO] [--id_in_background IDS...] [--only_desk+background] [--after_inpaint] -s SOURCE_1 -m MODEL_1 [--support_object_ids SUPPORT_IDS_1] [--object_postprocess_skip_labels SKIP_LABELS_1] [-s SOURCE_2 -m MODEL_2 [--support_object_ids SUPPORT_IDS_2] [--object_postprocess_skip_labels SKIP_LABELS_2] ...]

Examples:
  # Use DEFAULT_PAIRS defined inside this file:
  bash pipeline_wrapper.sh

  bash pipeline_wrapper.sh \
    --pair "../data/inpaint360/scene_a" "../output/scene_a/cobgs/4230000" "3 5 7" \
    --pair "../data/inpaint360/scene_b" "../output/scene_b/cobgs/4230000" "2,4,8"

  # Backward-compatible form:
  bash pipeline_wrapper.sh \
    -s "../data/inpaint360/scene_a" -m "../output/scene_a/cobgs/4230000" --support_object_ids "3 5 7" --object_postprocess_skip_labels 255 \
    -s "../data/inpaint360/scene_b" -m "../output/scene_b/cobgs/4230000" --support_object_ids "2,4,8" --object_postprocess_skip_labels 255
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/pipeline.sh"
REMOVAL_GT=0
REMOVAL_GT_RATIO=""
ID_IN_BACKGROUND=()
ONLY_DESK_BACKGROUND=0
AFTER_INPAINT=0

# Fill this list when you want to launch multiple scenes without passing pairs
# from the terminal. Each entry is "SOURCE_PATH|MODEL_PATH|SUPPORT_OBJECT_IDS|OBJECT_POSTPROCESS_SKIP_LABELS".
# The third field is optional; leave it empty to let export_desk_atlas.py use its default support ids.
# The fourth field is optional; use values like "255", "17 255", or "17,255" to skip postprocess for those labels.
DEFAULT_PAIRS=(
  # "../data/bear|../output/bear/cobgs/4280600|128|255"
  # "../data/bonsai|../output/bonsai/cobgs/4280600|128|255"
  # "../data/office_desk|../output/office_desk/cobgs/5010000|17 34 51 68 85 102 119|136 255"
  "../data/dining_table|../output/dining_table/cobgs/5010000|85 102 119 136 153 170 187 204|17 34 51 68 255"
  # "../data/bedroom|../output/bedroom/cobgs/5010000|17 34 51 68 85 102 136 153|255"
  # "../data/scene_1_colmap|../output/scene_1_colmap/cobgs/5080000|12 25 38 51 63 76|255"
  # "../data/scene_5_colmap|../output/scene_5_colmap/cobgs/5080000|12 25 38 51 63 76|255"
  # "../data/scene_6_colmap|../output/scene_6_colmap/cobgs/5080000|12 25 38 51 63 76|255"
)

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "Missing pipeline script: $PIPELINE_SCRIPT" >&2
  exit 1
fi

SOURCES=()
MODELS=()
SUPPORT_OBJECT_IDS=()
OBJECT_POSTPROCESS_SKIP_LABELS=()
PENDING_SOURCE=""
PENDING_MODEL=""
PENDING_SUPPORT_OBJECT_IDS=""
PENDING_OBJECT_POSTPROCESS_SKIP_LABELS=""

add_pair() {
  local source_path="${1%/}"
  local model_path="${2%/}"
  local support_object_ids="${3:-}"
  local object_postprocess_skip_labels="${4:-}"
  if [[ -z "$source_path" || -z "$model_path" ]]; then
    echo "Empty source_path or model_path is not allowed." >&2
    exit 1
  fi
  SOURCES+=("$source_path")
  MODELS+=("$model_path")
  SUPPORT_OBJECT_IDS+=("$support_object_ids")
  OBJECT_POSTPROCESS_SKIP_LABELS+=("$object_postprocess_skip_labels")
}

add_pair_string() {
  local pair="$1"
  local source_path=""
  local model_path=""
  local support_object_ids=""
  local object_postprocess_skip_labels=""
  if [[ "$pair" != *"|"* ]]; then
    echo "Invalid pair '$pair'. Expected format: SOURCE_PATH|MODEL_PATH[|SUPPORT_OBJECT_IDS[|OBJECT_POSTPROCESS_SKIP_LABELS]]" >&2
    exit 1
  fi
  IFS='|' read -r source_path model_path support_object_ids object_postprocess_skip_labels <<< "$pair"
  add_pair "$source_path" "$model_path" "$support_object_ids" "$object_postprocess_skip_labels"
}

flush_pending_pair() {
  if [[ -n "$PENDING_SOURCE" && -n "$PENDING_MODEL" ]]; then
    add_pair "$PENDING_SOURCE" "$PENDING_MODEL" "$PENDING_SUPPORT_OBJECT_IDS" "$PENDING_OBJECT_POSTPROCESS_SKIP_LABELS"
    PENDING_SOURCE=""
    PENDING_MODEL=""
    PENDING_SUPPORT_OBJECT_IDS=""
    PENDING_OBJECT_POSTPROCESS_SKIP_LABELS=""
    return
  fi
  if [[ -n "$PENDING_SOURCE" || -n "$PENDING_MODEL" || -n "$PENDING_SUPPORT_OBJECT_IDS" || -n "$PENDING_OBJECT_POSTPROCESS_SKIP_LABELS" ]]; then
    echo "Incomplete pair: source_path and model_path are required before starting another pair." >&2
    exit 1
  fi
}

set_pending_source() {
  if [[ -n "$PENDING_SOURCE" ]]; then
    flush_pending_pair
  fi
  PENDING_SOURCE="$1"
}

set_pending_model() {
  if [[ -n "$PENDING_MODEL" ]]; then
    flush_pending_pair
  fi
  PENDING_MODEL="$1"
}

set_pending_support_object_ids() {
  if [[ -z "$PENDING_SOURCE" && -z "$PENDING_MODEL" ]]; then
    echo "--support_object_ids must follow a pending -s/--source_path and -m/--model_path pair." >&2
    exit 1
  fi
  if [[ -n "$PENDING_SUPPORT_OBJECT_IDS" ]]; then
    echo "Got duplicate --support_object_ids for the same pair: $PENDING_SUPPORT_OBJECT_IDS" >&2
    exit 1
  fi
  PENDING_SUPPORT_OBJECT_IDS="$1"
}

set_pending_object_postprocess_skip_labels() {
  if [[ -z "$PENDING_SOURCE" && -z "$PENDING_MODEL" ]]; then
    echo "--object_postprocess_skip_labels must follow a pending -s/--source_path and -m/--model_path pair." >&2
    exit 1
  fi
  if [[ -n "$PENDING_OBJECT_POSTPROCESS_SKIP_LABELS" ]]; then
    echo "Got duplicate --object_postprocess_skip_labels for the same pair: $PENDING_OBJECT_POSTPROCESS_SKIP_LABELS" >&2
    exit 1
  fi
  PENDING_OBJECT_POSTPROCESS_SKIP_LABELS="$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    -s|--source_path|--source-path)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      set_pending_source "$2"
      shift 2
      ;;
    --source_path=*|--source-path=*)
      set_pending_source "${1#*=}"
      shift
      ;;
    -m|--model_path|--model-path)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      set_pending_model "$2"
      shift 2
      ;;
    --model_path=*|--model-path=*)
      set_pending_model "${1#*=}"
      shift
      ;;
    --support_object_ids|--support-object-ids)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 1
      fi
      shift
      support_values=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        support_values+=("$1")
        shift
      done
      if [[ ${#support_values[@]} -eq 0 ]]; then
        echo "Missing value for --support_object_ids" >&2
        usage >&2
        exit 1
      fi
      set_pending_support_object_ids "${support_values[*]}"
      ;;
    --support_object_ids=*|--support-object-ids=*)
      set_pending_support_object_ids "${1#*=}"
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
      set_pending_object_postprocess_skip_labels "${skip_values[*]}"
      ;;
    --object_postprocess_skip_labels=*|--object-postprocess-skip-labels=*)
      set_pending_object_postprocess_skip_labels "${1#*=}"
      shift
      ;;
    --pair)
      if [[ $# -lt 3 ]]; then
        echo "Missing SOURCE_PATH and MODEL_PATH for --pair" >&2
        usage >&2
        exit 1
      fi
      if [[ -n "$PENDING_SOURCE" || -n "$PENDING_MODEL" ]]; then
        flush_pending_pair
      fi
      source_path="$2"
      model_path="$3"
      shift 3
      support_values=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        support_values+=("$1")
        shift
      done
      add_pair "$source_path" "$model_path" "${support_values[*]}" ""
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

flush_pending_pair

if [[ ${#SOURCES[@]} -eq 0 ]]; then
  if [[ ${#DEFAULT_PAIRS[@]} -eq 0 ]]; then
    echo "No command-line pairs were provided and DEFAULT_PAIRS is empty." >&2
    usage >&2
    exit 1
  fi
  echo "No command-line pairs provided; using DEFAULT_PAIRS from pipeline_wrapper.sh."
  for pair in "${DEFAULT_PAIRS[@]}"; do
    add_pair_string "$pair"
  done
fi

cd "$SCRIPT_DIR"

for idx in "${!SOURCES[@]}"; do
  run_id=$((idx + 1))
  total=${#SOURCES[@]}
  source_path="${SOURCES[$idx]}"
  model_path="${MODELS[$idx]}"
  support_object_ids="${SUPPORT_OBJECT_IDS[$idx]}"
  object_postprocess_skip_labels="${OBJECT_POSTPROCESS_SKIP_LABELS[$idx]}"
  removal_gt_status="disabled"
  if [[ "$REMOVAL_GT" -eq 1 ]]; then
    removal_gt_status="enabled"
  fi
  only_desk_background_status="disabled"
  if [[ "$ONLY_DESK_BACKGROUND" -eq 1 ]]; then
    only_desk_background_status="enabled"
  fi
  after_inpaint_status="disabled"
  if [[ "$AFTER_INPAINT" -eq 1 ]]; then
    after_inpaint_status="enabled"
  fi

  echo "============================================================"
  echo "Pipeline run ${run_id}/${total}"
  echo "source_path: $source_path"
  echo "model_path : $model_path"
  echo "support_object_ids: ${support_object_ids:-<default>}"
  echo "object_postprocess_skip_labels: ${object_postprocess_skip_labels:-<none>}"
  echo "removal_GT: $removal_gt_status"
  echo "removal_GT_ratio: ${REMOVAL_GT_RATIO:-<default>}"
  echo "id_in_background: ${ID_IN_BACKGROUND[*]:-<none>}"
  echo "only_desk+background: $only_desk_background_status"
  echo "after_inpaint metrics: $after_inpaint_status"
  echo "============================================================"

  pipeline_args=(-s "$source_path" -m "$model_path")
  if [[ "$REMOVAL_GT" -eq 1 ]]; then
    pipeline_args+=(--removal_GT)
  fi
  if [[ -n "$REMOVAL_GT_RATIO" ]]; then
    pipeline_args+=(--removal_GT_ratio "$REMOVAL_GT_RATIO")
  fi
  if [[ ${#ID_IN_BACKGROUND[@]} -gt 0 ]]; then
    pipeline_args+=(--id_in_background "${ID_IN_BACKGROUND[@]}")
  fi
  if [[ "$ONLY_DESK_BACKGROUND" -eq 1 ]]; then
    pipeline_args+=(--only_desk+background)
  fi
  if [[ "$AFTER_INPAINT" -eq 1 ]]; then
    pipeline_args+=(--after_inpaint)
  fi
  if [[ -n "$support_object_ids" ]]; then
    pipeline_args+=(--support_object_ids "$support_object_ids")
  fi
  if [[ -n "$object_postprocess_skip_labels" ]]; then
    pipeline_args+=(--object_postprocess_skip_labels "$object_postprocess_skip_labels")
  fi
  bash "$PIPELINE_SCRIPT" "${pipeline_args[@]}"
done
