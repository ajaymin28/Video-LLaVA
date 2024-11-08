#!/bin/bash

DOWN_MODEL_PATH=$1 ##url="https://huggingface.co/ajaymin28/vl-sg-AG-fulltune/resolve/main/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_AG_v5_3_split0_bash_fulltune"
folder_name=$(basename "$DOWN_MODEL_PATH")
OUTPUT_DIR="$2/$folder_name/videollava-7b-lora"

mkdir -p $OUTPUT_DIR
wget -O "$OUTPUT_DIR/adapter_config.json" $DOWN_MODEL_PATH/videollava-7b-lora/adapter_config.json
wget -O "$OUTPUT_DIR/adapter_model.bin" $DOWN_MODEL_PATH/videollava-7b-lora/adapter_model.safetensors
wget -O "$OUTPUT_DIR/config.json" $DOWN_MODEL_PATH/videollava-7b-lora/config.json
wget -O "$OUTPUT_DIR/non_lora_trainables.bin" $DOWN_MODEL_PATH/videollava-7b-lora/non_lora_trainables.bin
wget -O "$OUTPUT_DIR/special_tokens_map.json" $DOWN_MODEL_PATH/videollava-7b-lora/special_tokens_map.json
wget -O "$OUTPUT_DIR/tokenizer_config.json" $DOWN_MODEL_PATH/videollava-7b-lora/tokenizer_config.json
wget -O "$OUTPUT_DIR/tokenizer.model" $DOWN_MODEL_PATH/videollava-7b-lora/tokenizer.model
wget -O "$OUTPUT_DIR/trainer_state.json" $DOWN_MODEL_PATH/videollava-7b-lora/trainer_state.json


## Usage: bash /home/jbhol/dso/gits/Video-LLaVA/scripts/download_models/download_HF_LORA_model.sh https://huggingface.co/ajaymin28/vl-1.5-AG-lora/resolve/main/%5Blora%5Dvideo_llava_AG_annotations_v5_3_p01_e01/videollava-7b-lora /groups/sernam/VideoSGG/checkpoints/llava_vicuna/AG