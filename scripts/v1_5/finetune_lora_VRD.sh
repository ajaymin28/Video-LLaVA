#!/bin/bash
JSON_FOLDER="/home/jbhol/dso/gits/LLaVA-NeXT/data_prep/data/video_llava_vidvrd_annotations_v5_3_shuffled"
IMAGE_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos"
VIDEO_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos"
cd /home/jbhol/dso/gits/Video-LLaVA

##${JSON_FOLDER}/videochatgpt_tune_part1.json ${JSON_FOLDER}/videochatgpt_tune_part2.json ${JSON_FOLDER}/videochatgpt_tune_part3.json ${JSON_FOLDER}/videochatgpt_tune_part4.json ${JSON_FOLDER}/videochatgpt_tune_part5.json ${JSON_FOLDER}/videochatgpt_tune_part6.json ${JSON_FOLDER}/videochatgpt_tune_part7.json ${JSON_FOLDER}/videochatgpt_tune_part8.json ${JSON_FOLDER}/videochatgpt_tune_part9.json ${JSON_FOLDER}/videochatgpt_tune_part10.json ${JSON_FOLDER}/videochatgpt_tune_part11.json ${JSON_FOLDER}/videochatgpt_tune_part12.json 
##${JSON_FOLDER}/videochatgpt_tune_part1.json
## ${JSON_FOLDER}/videochatgpt_tune_part1.json ${JSON_FOLDER}/videochatgpt_tune_part2.json ${JSON_FOLDER}/videochatgpt_tune_part3.json ${JSON_FOLDER}/videochatgpt_tune_part4.json ${JSON_FOLDER}/videochatgpt_tune_part5.json ${JSON_FOLDER}/videochatgpt_tune_part6.json ${JSON_FOLDER}/videochatgpt_tune_part7.json ${JSON_FOLDER}/videochatgpt_tune_part8.json ${JSON_FOLDER}/videochatgpt_tune_part9.json
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 deepspeed  --master_port 24223  "videollava/train/train_mem.py" \
    --lora_enable True --lora_r 256 --lora_alpha 512 --mm_projector_lr 2e-5 \
    --deepspeed /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path  ${JSON_FOLDER}/videochatgpt_tune_part0.json ${JSON_FOLDER}/videochatgpt_tune_part1.json ${JSON_FOLDER}/videochatgpt_tune_part2.json ${JSON_FOLDER}/videochatgpt_tune_part3.json ${JSON_FOLDER}/videochatgpt_tune_part4.json ${JSON_FOLDER}/videochatgpt_tune_part5.json ${JSON_FOLDER}/videochatgpt_tune_part6.json ${JSON_FOLDER}/videochatgpt_tune_part7.json ${JSON_FOLDER}/videochatgpt_tune_part8.json ${JSON_FOLDER}/videochatgpt_tune_part9.json   \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /home/jbhol/dso/gits/Video-LLaVA/checkpoints/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/jbhol/dso/gits/Video-LLaVA/checkpoints/vrd/[lora]video_llava_VRD_annotations_v5_3_p09_e01/videollava-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072  \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/cache_dir \
    --save_tokenizer True \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --init_lora_weights olora \
    # --tune_mm_mlp_adapter False \
    # --freeze_mm_mlp_adapter True \
    # --train_video_tower False \