CUDA_VISIBLE_DEVICES=0 python /home/jiangran/SegEarth-R2/SegEarth-R2/segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path=/home/jiangran/SegEarth-R2/SegEarth-R2/scripts/output/output_folder_8/checkpoint-10000 \
    --vision_tower=/home/jiangran/SegEarth-R2/pretrained_model/CLIP/siglip-so400m-patch14-384 \
    --vision_tower_mask=/home/jiangran/SegEarth-R2/pretrained_model/mask2former/model_final_54b88a.pkl \
    --mask_config=/home/jiangran/SegEarth-R2/SegEarth-R2/segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
    --save_path=./output/output_merged_segearth_r2_model_8 \
    --lora_r=4