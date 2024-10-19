#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=e02_vidor_triplets_llava_onevision_finetune
#SBATCH --time=3-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu

export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib

module load gcc/gcc-11.2.0

# llava one vision v6 prompt
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevision.py --output_dir=llava_vidvrd_onevision_v6  --conv-mode=qwen_2


# llava one vision v7 prompt
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevision.py --output_dir=llava_vidvrd_onevision_v7  --conv-mode=qwen_2

# llava one vision v8 prompt
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevision.py --output_dir=llava_vidvrd_onevision_v8  --conv-mode=qwen_2

# llava one vision v9 prompt
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevision.py --output_dir=llava_vidvrd_onevision_v9_30predicates  --conv-mode=qwen_2

# v11 time prompt
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v5_2_onevision.py --output_dir=pvsg_timeframe --conv-mode=qwen_2

###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_withTemporal.py --output_dir=pvsg_timeframe_v13 --conv-mode=qwen_2


###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_withTemporal.py --output_dir=pvsg_tempsanity --conv-mode=qwen_2


###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_withID_temporal.py --output_dir=pvsg_v13_withid_temporal --conv-mode=qwen_2

## plain v13 without id
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_plain.py --output_dir=pvsg_v13_woid_vidkeyword --conv-mode=qwen_2


## plain v13 with id

##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_withID.py --output_dir=pvsg_v13_withid_gpt --conv-mode=qwen_2

## fixed number of words for description 100-200
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_withID.py --output_dir=pvsg_v13_withid_gptindexedlist_200word --conv-mode=qwen_2

# vidor triplets without objects list
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_withID.py --output_dir=pvsg_v14_withid_woobj_list_triplets --conv-mode=qwen_2

# vidor quad without objects list
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_withID_Quad.py --output_dir=pvsg_v14_withid_woobj_list_quad --conv-mode=qwen_2



## vidvrd
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevisionWithID_ZS.py --output_dir=vidvrd_v13_withid_gptindexedlist_200word --conv-mode=qwen_2
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevisionWithID_ZS.py --output_dir=vidvrd_v13_indexedlist --conv-mode=qwen_2

###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevisionWithID_ZS.py --output_dir=vidvrd_v13_indexedlist3 --conv-mode=qwen_2

###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_onevisionWithID_ZS_Quad.py --output_dir=vidvrd_v14_withid_quad_customprompt --conv-mode=qwen_2
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_v13_onevision_plain.py --output_dir=pvsg_v13_woid_sam --conv-mode=qwen_2


## Llava onevision finetune epoch 1 
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_3_onevision_finetune.py --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_run2 --conv-mode=qwen_2

## Llava onevision finetune epoch 2
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_3_onevision_finetune.py --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e02 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_e02 --conv-mode=qwen_2

## Llava onevision finetune epoch 3
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_3_onevision_finetune.py --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e03 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_e03 --conv-mode=qwen_2


## Llava onevision finetune epoch 1 vidor zs test
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_zs_onevision_finetuned.py --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3 --output_dir=[test][onevision]_vidor_zs_e01 --conv-mode=qwen_2

## Llava onevision finetune epoch 3 vidor zs test
CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_zs_onevision_finetuned.py --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e02 --output_dir=[test][onevision]_vidor_zs_e02 --conv-mode=qwen_2