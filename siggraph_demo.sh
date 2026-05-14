cd /home/kunxinguang/Sourcecode/3D-Seg/GROUPING/siga26/mod-COB-GS

SOURCE_PATH=../data/bonsai
MODEL_PATH=../output/bonsai/cobgs/4280600
ITER=35000
DESK_ID=255

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "../data/bonsai" \
  -m "../output/bonsai/cobgs/4280600" \
  --iteration 35000 \
  --include_mask \
  --mask_mode multi_label \
  --desk_object_id "255" \
  --segmentation_checkpoint "../output/bonsai/cobgs/4280600/point_cloud/iteration_35000/point_cloud.pth" \
  --only_desk+background \
  --render_path \
  --render_path_frames 240 \
  --render_path_fps 30 \
  --skip_train \
  --skip_test \
  --eval \


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "../data/figurines" \
  -m "../output/figurines/cobgs/4210200" \
  --iteration 35000 \
  --include_mask \
  --mask_mode multi_label \
  --desk_object_id "255" \
  --segmentation_checkpoint "../output/figurines/cobgs/4210200/point_cloud/iteration_35000/point_cloud.pth" \
  --only_desk+background \
  --render_path \
  --render_path_frames 240 \
  --render_path_fps 30 \
  --skip_train \
  --skip_test \
  --eval \

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "../data/doppelherz" \
  -m "../output/doppelherz/cobgs/4230000" \
  --iteration 35000 \
  --include_mask \
  --mask_mode multi_label \
  --desk_object_id "255" \
  --segmentation_checkpoint "../output/doppelherz/cobgs/4230000/point_cloud/iteration_35000/point_cloud.pth" \
  --only_desk+background \
  --render_path \
  --render_path_frames 240 \
  --render_path_fps 30 \
  --skip_train \
  --skip_test \
  --eval \

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "../data/fruits" \
  -m "../output/fruits/cobgs/4230000" \
  --iteration 35000 \
  --include_mask \
  --mask_mode multi_label \
  --desk_object_id "255" \
  --segmentation_checkpoint "../output/fruits/cobgs/4230000/point_cloud/iteration_35000/point_cloud.pth" \
  --only_desk+background \
  --render_path \
  --render_path_frames 240 \
  --render_path_fps 30 \
  --skip_train \
  --skip_test \
  --eval \


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "../data/dining_table" \
  -m "../output/dining_table/cobgs/5010000" \
  --iteration 35000 \
  --include_mask \
  --mask_mode multi_label \
  --desk_object_id "255" \
  --segmentation_checkpoint "../output/dining_table/cobgs/5010000/point_cloud/iteration_35000/point_cloud.pth" \
  --only_desk+background \
  --render_path \
  --render_path_frames 240 \
  --render_path_fps 30 \
  --skip_train \
  --skip_test \
  --eval \
  --id_in_background 17 34 51 68 85 102 119 136 153 170 187 \


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python render.py \
  -s "../data/bedroom" \
  -m "../output/bedroom/cobgs/5010000" \
  --iteration 35000 \
  --include_mask \
  --mask_mode multi_label \
  --desk_object_id "255" \
  --segmentation_checkpoint "../output/bedroom/cobgs/5010000/point_cloud/iteration_35000/point_cloud.pth" \
  --only_desk+background \
  --render_path \
  --render_path_frames 240 \
  --render_path_fps 30 \
  --skip_train \
  --skip_test \
  --eval \
  --id_in_background 17 34 51 68 85 \