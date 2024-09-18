import argparse
import os

import torch

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import load_image, image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollava.constants import SGSpecialTokens

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import numpy as np
import cv2

from typing import Dict
import transformers

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_default_indices(video_path, frames_to_add=8):
    total_frames = getVideoFrameCount(video_path=video_path)
    if total_frames is not None:
        return np.linspace(0, total_frames-1, frames_to_add, dtype=int)
    else:
        return np.array([i for i in range(frames_to_add)])

    
def getVideoFrameCount(video_path):
    cv2_vr = cv2.VideoCapture(video_path)
    total_frames = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2_vr.release()
    return total_frames

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit,
                                                                     device=args.device, cache_dir=args.cache_dir)

    # num_added_toks = tokenizer.add_tokens(['<frameseg>'], special_tokens=True) ##This line is updated
    # model.resize_token_embeddings(len(tokenizer))
    
    ## Jaimin Changes
    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=dict(additional_special_tokens=["<frameseg>"]),
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    print(tokenizer.all_special_tokens)

    image_processor, video_processor = processor['image'], processor['video']
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    print(args.conv_mode)
    
    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    tensor = []
    special_token = []
    frame_indices = [i for i in range(0,8)]
    # frame_indices = [8,9,10,20,30,40,50,60,70,80]
    args.file = args.file if isinstance(args.file, list) else [args.file]
    for file in args.file:
        if os.path.splitext(file)[-1].lower() in image_ext:
            file = image_processor.preprocess(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN]
        elif os.path.splitext(file)[-1].lower() in video_ext:
            
            #file = video_processor(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            # video = [video_processor(i, return_tensors='pt',frame_indices=frame_indices,total_frames=total_frames)['pixel_values'][0] for i in video]  # fake image
            """Jaimin Changes: """
            file = video_processor(file, return_tensors='pt',frame_indices=frame_indices,total_frames=getVideoFrameCount(args.file[0]))['pixel_values'][0].to(model.device, dtype=torch.float16)
            # special_token += [f"{SGSpecialTokens.VIDEO_FRAME_ID}{DEFAULT_IMAGE_TOKEN}"] * model.get_video_tower().config.num_frames
            special_token += [f"{DEFAULT_IMAGE_TOKEN}"] * model.get_video_tower().config.num_frames
        else:
            raise ValueError(f'Support video of {video_ext} and image of {image_ext}, but found {os.path.splitext(file)[-1].lower()}')
        print(file.shape)
        print("frames=>", frame_indices)
        tensor.append(file)

    print(special_token)
    isFrameReset = False
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
            
        if inp=="SGG":
            inp = """Please follow these steps for the given video. 
                  Step 1: Generate Scene Graph with the following format [subject,predicate,object]. 
                  Step 2: Provide bounding box locations for each subject and Object in the format [x_min,y_min,x_max,y_max].
                  
                  Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb.
                  """

        if "setframes" in inp:
            print("setting frames...")
            try:
                inp = inp.split("=>")[-1]
                framerange = eval(inp.strip())
                temp_frame_indices = [i for i in range(int(framerange[0]),int(framerange[1]))]
                if len(temp_frame_indices)>8:
                    temp_frame_indices = temp_frame_indices[0:8]
                    print("only 8 frames length is supported, using => ", temp_frame_indices)
                frame_indices = temp_frame_indices
                inp="reset"
                isFrameReset = True
            except Exception as e:
                print("e", e)
                continue

        
        if inp=="reset":
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            tensor = []
            special_token = []
            args.file = args.file if isinstance(args.file, list) else [args.file]
            for file in args.file:
                if os.path.splitext(file)[-1].lower() in image_ext:
                    file = image_processor.preprocess(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
                    special_token += [DEFAULT_IMAGE_TOKEN]
                elif os.path.splitext(file)[-1].lower() in video_ext:
                    
                    #file = video_processor(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
                    """Jaimin Changes: """
                    # video = [video_processor(i, return_tensors='pt',frame_indices=frame_indices,total_frames=total_frames)['pixel_values'][0] for i in video]  # fake image
                    file = video_processor(file, return_tensors='pt',frame_indices=frame_indices,total_frames=getVideoFrameCount(args.file[0]))['pixel_values'][0].to(model.device, dtype=torch.float16)
                    # special_token += [f"{SGSpecialTokens.VIDEO_FRAME_ID}{DEFAULT_IMAGE_TOKEN}"] * model.get_video_tower().config.num_frames
                    special_token += [f"{DEFAULT_IMAGE_TOKEN}"] * model.get_video_tower().config.num_frames
                else:
                    raise ValueError(f'Support video of {video_ext} and image of {image_ext}, but found {os.path.splitext(file)[-1].lower()}')
                print(file.shape)
                print("frames=>", frame_indices)
                tensor.append(file)
                # file = None  # we dont need to add frames tokens to conv again, features are taken in tensor which will be passed to model.

            print("Reset complete")
            continue

        print(f"{roles[1]}: ", end="")

        if file is not None:
            # first message
            if getattr(model.config, "mm_use_im_start_end", False):
                inp = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + '\n' + inp
            else:
                inp = ''.join(special_token) + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            file = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        # print("input ids ",input_ids.shape)
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,  # video as fake images
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/jbhol/dso/gits/Video-LLaVA/checkpoints/video_llava_annotations_v16_e10/videollava-7b-lora")
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--file", nargs='+', type=str,default="/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos/0018_4748191834.mp4", required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2500)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
