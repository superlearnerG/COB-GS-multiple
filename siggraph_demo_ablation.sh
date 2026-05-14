python inpaint_ablation.py \
  -s ../data/bedroom \
  -m ../output/bedroom/cobgs/5010000 \
  --source_iteration 30000 \
  --iterations 32000 \
  --segmentation_checkpoint "../output/bedroom/cobgs/5010000/multi_object(ours)/final_multi_object.pth" \
  --preserve_object_id 255  \
  --id_in_background 255  \
  --only_desk+background \
  --supervision_path ../output/bedroom/cobgs/5010000/ablation/2d_inpainting_result_102_153_136_68_51_85_34_17 \
  --eval



SOURCE_PATH="$(realpath -m ../data/bedroom)"
BASE_MODEL="$(realpath -m ../output/bedroom/cobgs/5010000)"
ABLATION_MODEL="$BASE_MODEL/ablation"
ITER=32000

ln -sfnT "$BASE_MODEL/cfg_args" "$ABLATION_MODEL/cfg_args"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "$SOURCE_PATH" \
  -m "$ABLATION_MODEL" \
  --iteration "$ITER" \
  --include_mask \
  --mask_mode multi_label \
  --desk_object_id 255 \
  --segmentation_checkpoint "$ABLATION_MODEL/point_cloud/iteration_${ITER}/point_cloud.pth" \
  --id_in_background 255 \
  --only_desk+background \
  --render_path \
  --render_path_frames 240 \
  --render_path_fps 30 \
  --skip_train \
  --skip_test \
  --eval
