

# JSON_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v12"
# IMAGE_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos"
# VIDEO_FOLDER="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos"
# cd /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/

## vidvrd
JSON_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v2_1_nomax"
IMAGE_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos"
VIDEO_FOLDER="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos"

cd /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA


module load cuda/cuda-12.1
module lodd gcc/gcc-11.2.0

#python "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/OPVSG_VIdeoLLAVA_Annot_chatgpt_v9.py"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --master_port 29300 videollava/train/train_mem.py \
    --deepspeed /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/videochatgpt_tune_part0.json ${JSON_FOLDER}/videochatgpt_tune_part1.json ${JSON_FOLDER}/videochatgpt_tune_part2.json ${JSON_FOLDER}/videochatgpt_tune_part3.json ${JSON_FOLDER}/videochatgpt_tune_part4.json ${JSON_FOLDER}/videochatgpt_tune_part5.json \
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
    --output_dir /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/finetune/finetune_video_llava_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --cache_dir /lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/cache_dir
