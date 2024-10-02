#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=t04_v5_3_customchange_e03_video_llava_vidvrd_annotations_temp04
#SBATCH --time=2-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu

export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib

module load gcc/gcc-11.2.0

## 5k data train evaluation on vidvrd test epochs 5
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test]closevocab_video_llava_vidvrd_annotations_v3_1_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v3_1_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

# v4_2 P05E05
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v4_2_nomax_p05_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v4_2_nomax_p05_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


## 5k data train evaluation on vidvrd train epochs 5
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1_train.py" --output_dir=[train]video_llava_vidvrd_annotations_v3_1_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v3_1_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


# v5_1 P09_E01 with sys prompt
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3_1_with_sys_prompt.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_1_withsysmsg_p09_e01_withupdated_prompt --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]alpha256_video_llava_vidvrd_annotations_v5_1_withsysmsg_p09_e01/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


# v5_2 P09_E03 Temprature 0.2
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_varyingprompts.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_3_e03_2 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_customchange_e03/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --file="" --temperature=0.2

# v5_2 P09_E03 Temprature 0.0
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_varyingprompts.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_3_e03_2_t00 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_customchange_e03/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --file="" --temperature=0.0

# v5_2 P09_E03 Temprature 0.4
CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_2_varyingprompts.py" --output_dir=[test][lora]alpha256_video_llava_vidvrd_annotations_v5_3_e03_2_t04 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/[lora]video_llava_vidvrd_annotations_v5_2_withsysmsg_varying_list_customchange_e03/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --file="" --temperature=0.4