#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=Eval_projection_train_10_video_llava_vidvrd_annotations_v2_1_nomax_e10
#SBATCH --time=2-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu


export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib

module load gcc/gcc-11.2.0


## epoch 1 /home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax/videollava-7b-lora
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=[train]video_llava_vidvrd_annotations_v2_1_nomax_e01 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## epoch 5 
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=[train]10_video_llava_vidvrd_annotations_v2_1_nomax_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## epoch 10 
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=[train]10_video_llava_vidvrd_annotations_v2_1_nomax_e10 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_e10/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


## epoch 5 LORA 256 aplha 512
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=[train]lora256a512_video_llava_vidvrd_annotations_v2_1_nomax_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/lora256_alpha512/video_llava_vidvrd_annotations_v2_1_nomax/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## mlp tune projection e05
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=[train]projection_video_llava_vidvrd_annotations_v2_1_nomax_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/tunemlp/video_llava_mlptune_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## mlp tune projection e10
CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=[train]projection_video_llava_vidvrd_annotations_v2_1_nomax_e10 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/tunemlp/video_llava_vidvrd_annotations_v2_1_nomax_e10/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""




###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidor_v16_vid.py" --output_dir=v19_wobb_noseptoken_t0_vidor --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_annotations_v19_wobb_e02_nospectoken/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

###vidor with vidvrd model
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidor_v16_vid.py" --output_dir=vidor_video_llava_vidvrd_annotations_v2_1_nomax2 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train vidor eval #378424
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidor_v16_vid.py" --output_dir=vidor_video_llava_vidvrd_annotations_v2_1_part05_nomax --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on vidvrd
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=video_llava_vidvrd_annotations_v2_1_nomax_p05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on vidvrd train
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=train_video_llava_vidvrd_annotations_v2_1_nomax_p05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on vidvrd train MLP projection unzfrozen
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=train_video_llava_vidvrd_annotations_v2_1_nomax_p05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on vidvrd train MLP projection unzfrozen
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=train_video_llava_vidvrd_annotations_v2_1_nomax_p05_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on vidvrd train lora r256a512
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=train_video_llava_vidvrd_annotations_v2_1_nomax_p05_lorar256a512 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/lora256_alpha512/video_llava_vidvrd_annotations_v2_1_nomax/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on vidvrd test epochs 5
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=video_llava_vidvrd_annotations_v2_1_nomax_p05_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k data train evaluation on vidvrd train epochs 5
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_on_train.py" --output_dir=[train]video_llava_vidvrd_annotations_v2_1_nomax_p05_e05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_e05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


## 5k data train evaluation on vidvrd MLP projection unzfrozen
#CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=video_llava_mlptune_vidvrd_annotations_v2_1_nomax_p05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/tunemlp/video_llava_mlptune_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 5k V3 eval vidvrd () video_llava_without_underscore_vidvrd_annotations_v3_p05
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v3annot.py" --output_dir=video_llava_without_underscore_vidvrd_annotations_v3_p05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v3/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.2 --file=""

## 5k V3 eval vidor video_llava_without_underscore_vidvrd_annotations_v3_p05
##CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidor_v16_vid.py" --output_dir=vidor_video_llava_without_underscore_vidvrd_annotations_v3_p05 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_without_underscore_vidvrd_annotations_v3_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=virvrd_v2_part01_nomax_notokenizerchange --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/vidvrd_video_llava_annotations_vidvrd_v2_part01_e01_nospectoken_to_tokenizer/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=virvrd_v1_2_part0_10max --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/vidvrd_video_llava_annotations_v1_2_part0_10max/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## vidvrd full datatrain validation on vidvrd val data
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=vidvrd_video_llava_vidvrd_annotations_v2_1_nomax3 --model-path=/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

## 2k data train
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=tempdir --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/vidvrd_video_llava_annotations_vidvrd_v2_part01_e01_nospectoken/videollava-7b-lora/ --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


## 5k data train
###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd.py" --output_dir=video_llava_vidvrd_annotations_v2_1_nomax_p05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_vidvrd_annotations_v2_1_nomax_p05/videollava-7b-lora --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


###CUDA_VISIBLE_DEVICES=0 python "/home/jbhol/dso/gits/Video-LLaVA/vidvrd_cnt_gt.py" --output_dir=temp --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/vidvrd_video_llava_annotations_vidvrd_v2_part01_e01_nospectoken/videollava-7b-lora/ --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""


### epic_kitchen
# CUDA_VISIBLE_DEVICES=0 python "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/evaluate_epichkitchen_v16.py" --output_dir=v16_videoonly_t0_epickitchen --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_annotations_v16_e10/videollava-7b-lora/ --model-base lmsys/vicuna-7b-v1.5 --temperature=0.0 --file=""

###CUDA_VISIBLE_DEVICES=0 python "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidor.py" --output_dir=v12_2_videoonly_5_t_01 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/v12_2_videoonly_5/videollava-7b-lora/ --model-base lmsys/vicuna-7b-v1.5 --temperature=0.1 --file=""

###CUDA_VISIBLE_DEVICES=0 python "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidor.py" --output_dir=v12_2_videoonly_5_t_05 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/v12_2_videoonly_5/videollava-7b-lora/ --model-base lmsys/vicuna-7b-v1.5 --temperature=0.5 --file=""

###CUDA_VISIBLE_DEVICES=0 python "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidor.py" --output_dir=v12_2_videoonly_5_t_10 --model-path=/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/checkpoints/v12_2_videoonly_5/videollava-7b-lora/ --model-base lmsys/vicuna-7b-v1.5 --temperature=1.0 --file=""