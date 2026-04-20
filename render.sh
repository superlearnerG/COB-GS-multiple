PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -m "../output/figurines/cobgs/4141600" \
  --include_mask \
  --mask_mode multi_label \
  --N4views 22
