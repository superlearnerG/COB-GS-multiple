#!/usr/bin/env bash
set -euo pipefail

GROUP_OUTPUTS=()

run_metric() {
  local label="$1"
  shift

  local output_file
  output_file="$(mktemp)"
  GROUP_OUTPUTS+=("${output_file}")

  echo
  echo "===== ${label} ====="
  "$@" 2>&1 | tee "${output_file}"
}

print_group_average() {
  local group_name="$1"

  python - "${group_name}" "${GROUP_OUTPUTS[@]}" <<'PY'
import re
import sys
from pathlib import Path

group_name = sys.argv[1]
output_paths = [Path(path) for path in sys.argv[2:]]
metric_names = ("PSNR", "SSIM", "LPIPS", "FID")
pattern = re.compile(r"^\s*(PSNR|SSIM|LPIPS|FID)\s*:\s*([-+0-9.eE]+)\s*$")

values = {name: [] for name in metric_names}
for output_path in output_paths:
    metrics = {}
    for line in output_path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    missing = [name for name in metric_names if name not in metrics]
    if missing:
        raise SystemExit(f"{output_path} missing metrics: {missing}")
    for name in metric_names:
        values[name].append(metrics[name])

print("")
print(f"===== {group_name} average over {len(output_paths)} experiments =====")
for name in metric_names:
    mean_value = sum(values[name]) / len(values[name])
    print(f"  {name:<5}: {mean_value:>12.7f}")
PY

  rm -f "${GROUP_OUTPUTS[@]}"
  GROUP_OUTPUTS=()
}

run_metric "group1 bedroom" \
  python metrics_ablation.py \
    --input_path "../output/bedroom/cobgs/5010000/decouple+inpaint/只移走床+床头柜/test/render" \
    --GT_path "../data/bedroom/removal_GT_partial" \
    --compute_fid

run_metric "group1 dining_table" \
  python metrics_ablation.py \
    --input_path "../output/dining_table/cobgs/5010000/decouple+inpaint/desk+background_17_34_51_68_170_136_102_119_187_85_153/test/render" \
    --GT_path "../data/dining_table/removal_GT_partial" \
    --compute_fid

run_metric "group1 office_desk" \
  python metrics_ablation.py \
    --input_path "../output/office_desk/cobgs/5010000/decouple+inpaint/desk+background_136_119_102_17_85/test/render" \
    --GT_path "../data/office_desk/removal_GT_partial" \
    --compute_fid

print_group_average "group1"

run_metric "group2 office_desk" \
  python metrics_ablation.py \
    --input_path "../output/office_desk/cobgs/5010000/ablation/decouple+inpaint/desk+background_255_136_119_102_17_85/test/render" \
    --GT_path "../data/office_desk/removal_GT_partial" \
    --compute_fid

run_metric "group2 dining_table" \
  python metrics_ablation.py \
    --input_path "../output/dining_table/cobgs/5010000/ablation/decouple+inpaint/desk+background_255_17_34_51_68_170_136_102_119_187_85_153/test/render" \
    --GT_path "../data/dining_table/removal_GT_partial" \
    --compute_fid

run_metric "group2 bedroom" \
  python metrics_ablation.py \
    --input_path "../output/bedroom/cobgs/5010000/ablation/decouple+inpaint/desk+background_255_17_34_51_68_85/test/render" \
    --GT_path "../data/bedroom/removal_GT_partial" \
    --compute_fid

print_group_average "group2"

run_metric "group3 office_desk" \
  python metrics_ablation.py \
    --input_path "../output/office_desk/cobgs/5010000/ablation/decouple+inpaint/desk+background_255_136/test/render" \
    --GT_path "../data/office_desk/removal_GT" \
    --compute_fid

run_metric "group3 dining_table" \
  python metrics_ablation.py \
    --input_path "../output/dining_table/cobgs/5010000/ablation/decouple+inpaint/desk+background_255_17_34_51_68/test/render" \
    --GT_path "../data/dining_table/removal_GT" \
    --compute_fid

run_metric "group3 bedroom" \
  python metrics_ablation.py \
    --input_path "../output/bedroom/cobgs/5010000/ablation/decouple+inpaint/desk+background_255/test/render" \
    --GT_path "../data/bedroom/removal_GT" \
    --compute_fid

print_group_average "group3"
