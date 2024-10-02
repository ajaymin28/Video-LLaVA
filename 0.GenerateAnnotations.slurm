#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
###SBATCH --gres=gpu:1
###SBATCH --constraint=v100
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=v5_2_gen_annot_videollava_vidvrd_p09
#SBATCH --time=24:00:00
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

####python "/home/jbhol/dso/gits/OpenPVSG/OPVSG_VIdeoLLAVA_Annot_chatgpt_v19_wo_bb copy.py"
###python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava.py
##python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava_v3.py
####python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava_v3_1.py
###bash "/lustre/fs1/home/jbhol/dso/gits/Video-LLaVA/scripts/v1_5/finetune_lora.sh"
###python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava_v4.py
###python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava_v5.py
##python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava_v4.py
python /home/jbhol/dso/gits/VRDFormer_VRD/prepare_video_llava_v5_2.py
