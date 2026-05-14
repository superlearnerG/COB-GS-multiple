

for a in P0 P1 P2; do
  python export_desk_atlas_ablation.py \
    -s "../data/bedroom" \
    -m "../output/bedroom/cobgs/5010000" \
    --desk_object_id 255 \
    --support_object_ids 85 102 119 136 153 170 187 204 \
    --segmentation_checkpoint "../output/bedroom/cobgs/5010000/multi_object(ours)/final_multi_object.pth" \
    --desk_atlas_long_side 1254 \
    --desk_atlas_size_multiple 6 \
    --ccm_max_mask_samples 100000 \
    --ablation "$a"
done

for a in P0 P1 P2; do
  python export_desk_atlas_ablation.py \
    -s "../data/office_desk" \
    -m "../output/office_desk/cobgs/5010000" \
    --desk_object_id 255 \
    --support_object_ids 17 34 51 68 85 102 119 \
    --segmentation_checkpoint "../output/office_desk/cobgs/5010000/multi_object(ours)/final_multi_object.pth" \
    --desk_atlas_long_side 1254 \
    --desk_atlas_size_multiple 6 \
    --ccm_max_mask_samples 100000 \
    --ablation "$a"
done



# for a in F0 F1 F2 F3 F4; do
#   python export_desk_atlas_ablation.py \
#     -s "$DATASET_PATH" \
#     -m "$MODEL_PATH" \
#     --desk_object_id 255 \
#     --support_object_ids 85 102 119 136 153 170 187 204 \
#     --desk_atlas_long_side 1254 \
#     --desk_atlas_size_multiple 6 \
#     --ccm_max_mask_samples 1000000 \
#     --ablation "$a"
# done

