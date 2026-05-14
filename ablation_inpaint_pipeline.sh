# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint_ablation.py \
#   -s ../data/bedroom \
#   -m ../output/bedroom/cobgs/5010000 \
#   --source_iteration 30000 \
#   --iterations 32000 \
#   --segmentation_checkpoint ../output/bedroom/cobgs/5010000/multi_object/final_multi_object.pth \
#   --preserve_object_id 255 \
#   --id_in_background 255 \
#   --only_desk+background \
#   --supervision_path ../output/bedroom/cobgs/5010000/ablation/2d_inpainting_result_102_153_136_68_51_85_34_17 \
#   --eval


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint_ablation.py \
#   -s ../data/bedroom \
#   -m ../output/bedroom/cobgs/5010000 \
#   --source_iteration 30000 \
#   --iterations 32000 \
#   --segmentation_checkpoint ../output/bedroom/cobgs/5010000/multi_object/final_multi_object.pth \
#   --preserve_object_id 255 17 34 51 68 85 \
#   --id_in_background 255 17 34 51 68 85 \
#   --only_desk+background \
#   --supervision_path ../output/bedroom/cobgs/5010000/ablation/2d_inpainting_result_102_136_153 \
#   --eval



# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint_ablation.py \
#   -s ../data/dining_table \
#   -m ../output/dining_table/cobgs/5010000 \
#   --source_iteration 30000 \
#   --iterations 32000 \
#   --segmentation_checkpoint ../output/dining_table/cobgs/5010000/multi_object/final_multi_object.pth \
#   --preserve_object_id 255 17 34 51 68  \
#   --id_in_background 255 17 34 51 68  \
#   --only_desk+background \
#   --supervision_path ../output/dining_table/cobgs/5010000/ablation/2d_inpainting_result_204_170_136_102_119_187_85_153 \
#   --eval

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint_ablation.py \
#   -s ../data/dining_table \
#   -m ../output/dining_table/cobgs/5010000 \
#   --source_iteration 30000 \
#   --iterations 32000 \
#   --segmentation_checkpoint ../output/dining_table/cobgs/5010000/multi_object/final_multi_object.pth \
#   --preserve_object_id 255 17 34 51 68 170 136 102 119 187 85 153 \
#   --id_in_background 255 17 34 51 68 170 136 102 119 187 85 153 \
#   --only_desk+background \
#   --supervision_path ../output/dining_table/cobgs/5010000/ablation/2d_inpainting_result_204 \
#   --eval



PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint_ablation.py \
  -s ../data/office_desk \
  -m ../output/office_desk/cobgs/5010000 \
  --source_iteration 30000 \
  --iterations 32000 \
  --segmentation_checkpoint ../output/office_desk/cobgs/5010000/multi_object/final_multi_object.pth \
  --preserve_object_id 255 136 \
  --id_in_background 255 136 \
  --only_desk+background \
  --supervision_path ../output/office_desk/cobgs/5010000/ablation/2d_inpainting_result_119_102_17_34_51_68_85 \
  --eval

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python inpaint_ablation.py \
#   -s ../data/office_desk \
#   -m ../output/office_desk/cobgs/5010000 \
#   --source_iteration 30000 \
#   --iterations 32000 \
#   --segmentation_checkpoint ../output/office_desk/cobgs/5010000/multi_object/final_multi_object.pth \
#   --preserve_object_id 255 136 119 102 17 85 \
#   --id_in_background 255 136 119 102 17 85 \
#   --only_desk+background \
#   --supervision_path ../output/office_desk/cobgs/5010000/ablation/2d_inpainting_result_34_51_68 \
#   --eval