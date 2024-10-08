#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=v7_time_e01_varying_prompt_alpha256_video_llava_vidvrd_annotations
#SBATCH --time=1-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu

###Load modules


#module load anaconda/anaconda-2023.09
# conda init bash
#conda activate /lustre/fs1/home/jbhol/my-envs/vl12/

export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib
module load gcc/gcc-11.2.0

wandb login 8350cb094ed6779c41f439917838bbe922b6bec2
###python "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/OPVSG_VIdeoLLAVA_Annot_chatgpt_v18_5.py"
##bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"

## eval test data for p05_e05_v5
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=[test]video_llava_vidvrd_annotations_v5_nomax_p05_e05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha150_video_llava_vidvrd_annotations_v2_1_nomax_p05_e05/videollava-7b-lora --temperature=0.0 --file=""


## eval test data for p05_e05_v4_2
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=[test]v5_1_wosysmsg_video_llava_vidvrd_annotations_p05_e05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha150_video_llava_vidvrd_annotations_v5_1_wosysmsg_p05_e05/videollava-7b-lora --temperature=0.0 --file=""


## 5k data train evaluation
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=video_llava_vidvrd_annotations_v2_1_nomax_p05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on v5_1 annotation
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_1_withsysmsg_p05_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v5_1_withsysmsg_p05_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


# 9k data V2_1
#bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v2_1_nomax_p09_e03 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v2_1_nomax_p09_e03/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""



## 9k data train evaluation on v5_1 annotation
#bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_1_withsysmsg_p09_e01 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v5_1_withsysmsg_p09_e01/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


## v4_4
###bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v4_4_p09_e01 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v4_4_p09_e01/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


## v5_2 custom varying list
##bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_customchange --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_customchange/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""
###


## v5_2 custom varying list_2
##bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_2 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_2/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""
###


## v5_3 varying list
#python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava_v5_2.py
#bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_varyingprompts.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_3_e03 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v5_3_e03/videollava-7b-lora  --file=""


##v4_3
##bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora_v4_3.sh"
#Eval
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v4_3 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v4_3/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## v6
##bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
##/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v6_e03/videollava-7b-lora

## v6 ep01
##bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v6_varyingprompts.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v6_e01_fast --model-path="/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v6_e01/videollava-7b-lora" --model-base lmsys/vicuna-7b-v1.5 --file="" --temperature=0.2


##v7 ep03
#bash "/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lorav7.sh"
##/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v7_e03/videollava-7b-lora


#v7 with time
bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
##/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v7_withtime_e03/videollava-7b-lora \




