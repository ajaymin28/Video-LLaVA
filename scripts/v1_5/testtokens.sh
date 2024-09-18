#!/bin/bash
JSON_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v18_3"
IMAGE_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos"
VIDEO_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos"
cd /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA

## --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/videochatgpt_tune_.json ${JSON_FOLDER}/nlp_tune.json \
## --data_path ${JSON_FOLDER}/videochatgpt_tune_.json  \
##  ${JSON_FOLDER}/videochatgpt_tune_bb.json
## ${JSON_FOLDER}/videochatgpt_tune_.json

##python "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/OPVSG_VIdeoLLAVA_Annot_chatgpt_v9.py"
## ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/llava_image_tune_bb.json
# python /home/jbhol/dso/gits/OpenPVSG/OPVSG_VIdeoLLAVA_Annot_chatgpt_v18.py

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed  --master_port 29351  "/home/jbhol/dso/gits/Video-LLaVA/videollava/train/testtoken.py" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path  ${JSON_FOLDER}/videochatgpt_tune_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_annotations_v18_3_e01_spec_sg_token/videollava-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/cache_dir