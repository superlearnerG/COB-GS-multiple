# python simplelama_inpaint.py \
#   -s ../data/bedroom \
#   -m ../output/bedroom/cobgs/5010000 \
#   --image_dir ../output/bedroom/cobgs/5010000/decouple/desk+background/train/render/ \
#   --inpainting_mask 34 \
#   --output_dir ../output/bedroom/cobgs/5010000/ablation/_2d_inpainting_result_34 \
#   --device cuda \
#   --mask_dilation 10 


python simplelama_inpaint.py \
  -s ../data/bedroom \
  -m ../output/bedroom/cobgs/5010000 \
  --image_dir ../output/bedroom/cobgs/5010000/decouple/desk+background/train/render/ \
  --use_own_mask_path \
  --mask_path ../output/bedroom/cobgs/5010000/ablation/2d_inpainting_mask_102_153_136_68_51_85_34_17 \
  --output_dir ../output/bedroom/cobgs/5010000/ablation/2d_inpainting_result_102_153_136_68_51_85_34_17 \
  --device cuda \
  --mask_dilation 63