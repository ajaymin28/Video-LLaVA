#!/bin/bash
# JSON_FOLDER="/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v19_w_bb"
#JSON_FOLDER="/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v19_wo_bb"
#IMAGE_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos"
#VIDEO_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos"

## vidvrd
JSON_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v2_1_nomax"
##JSON_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v3_1"
#JSON_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v3"
IMAGE_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos"
VIDEO_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos"

cd /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA

## --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/videochatgpt_tune_.json ${JSON_FOLDER}/nlp_tune.json \
## --data_path ${JSON_FOLDER}/videochatgpt_tune_.json  \
##  ${JSON_FOLDER}/videochatgpt_tune_bb.json
## ${JSON_FOLDER}/videochatgpt_tune_.json

##python "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/OPVSG_VIdeoLLAVA_Annot_chatgpt_v9.py"
## ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/llava_image_tune_bb.json
# python /home/jbhol/dso/gits/OpenPVSG/OPVSG_VIdeoLLAVA_Annot_chatgpt_v18.py
#python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava.py

## b${JSON_FOLDER}/videochatgpt_tune_part0.json ${JSON_FOLDER}/videochatgpt_tune_part1.json ${JSON_FOLDER}/videochatgpt_tune_part2.json ${JSON_FOLDER}/videochatgpt_tune_part3.json ${JSON_FOLDER}/videochatgpt_tune_part4.json ${JSON_FOLDER}/videochatgpt_tune_part5.json ${JSON_FOLDER}/videochatgpt_tune_part6.json ${JSON_FOLDER}/videochatgpt_tune_part7.json ${JSON_FOLDER}/videochatgpt_tune_part8.json ${JSON_FOLDER}/videochatgpt_tune_part9.json

# /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/lora256_alpha512/video_llava_vidvrd_annotations_v2_1_nomax/videollava-7b-lora
# /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_e05/videollava-7b-lora

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed  --master_port 29313  "/home/jbhol/dso/gits/Video-LLaVA/videollava/train/train_mem.py" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path  ${JSON_FOLDER}/videochatgpt_tune_part0.json ${JSON_FOLDER}/videochatgpt_tune_part1.json ${JSON_FOLDER}/videochatgpt_tune_part2.json ${JSON_FOLDER}/videochatgpt_tune_part3.json ${JSON_FOLDER}/videochatgpt_tune_part4.json ${JSON_FOLDER}/videochatgpt_tune_part5.json ${JSON_FOLDER}/videochatgpt_tune_part6.json ${JSON_FOLDER}/videochatgpt_tune_part7.json ${JSON_FOLDER}/videochatgpt_tune_part8.json ${JSON_FOLDER}/videochatgpt_tune_part9.json  \
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
    --output_dir /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/tunemlp/video_llava_vidvrd_annotations_v2_1_nomax_p09_e10/videollava-7b-lora \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4000  --tokenizer_model_max_length 4000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/cache_dir \
    --save_tokenizer True \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    # --train_video_tower False \
    