import os
import json
import glob
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
import numpy as np
import time

from utils.utilities import create_batch_frames, eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_onevision_v6
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from myutils.utilities import getRandomPrompt

from tqdm import tqdm

import argparse
import os
import copy

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
# import openai
import pickle
# from PIL import Image

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

# from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN
# from videollava.conversation import conv_templates, SeparatorStyle
# from videollava.model.builder import load_pretrained_model
# from videollava.serve.utils import load_image, image_ext, video_ext
# from videollava.utils import disable_torch_init
# from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from videollava.constants import SGSpecialTokens

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import numpy as np
import cv2
import random

from typing import Dict
import transformers

from vidvrd2dataset import VidVRD, VidOR


def get_varying_list(current_block_list, full_list, fix_size=100):
	"""
	1. take current list (shuffle it)
	2. add elements to current list from full list without repeatation that sums to fix_size (shuffle it again)
	3. return the list
	"""
	current_block_list = set(copy.deepcopy(current_block_list))
	full_list = set(copy.deepcopy(full_list))

	newelements = full_list.difference(current_block_list)

	current_block_list = list(current_block_list)
	newelements =  list(newelements)
	newElementsNeeded = 0
	currentElementsSize = len(current_block_list) 
	if currentElementsSize>fix_size:
		## more items than predefined limit
		newElementsNeeded = 0
		pass
	else:
		newElementsNeeded = fix_size - len(current_block_list) 

	if len(newelements)<newElementsNeeded:
		current_block_list = current_block_list + random.sample(newelements,k=len(newelements))
	else:
		current_block_list = current_block_list + random.sample(newelements,k=newElementsNeeded)

	random.shuffle(current_block_list)
	return current_block_list

def get_default_indices(video_path, frames_to_add=8):
    total_frames = getVideoFrameCount(video_path=video_path)
    if total_frames is not None:
        return np.linspace(0, total_frames-1, frames_to_add, dtype=int)
    else:
        return np.array([i for i in range(frames_to_add)])


def set_video(args, video_frame_index=[0,1,2,3,4,5,6,7]):
    video_path = args.video_path
    global input_video, video_time, frame_time
    frame_idx = []
    # Check if the video exists
    if os.path.exists(video_path):
        if "gpt4v" != args.model_path:
            input_video,frame_time,video_time,frame_idx = load_video(video_path, args, video_frame_index=video_frame_index)
            input_video = image_processor.preprocess(input_video, return_tensors="pt")["pixel_values"].half().cuda()
            input_video = [input_video]
        else:
            spare_frames,frame_time,video_time,frame_idx = load_video_base64(video_path)
            interval = int(len(input_video) / args.for_get_frames_num)
    else:
        raise FileNotFoundError("Video file not found")

    return frame_idx

def handle_custom_commands(inp):
    actions = {
        "reset_loop": False,
        "exit_loop": False
    }
    if "setvideo" in inp:
        videoPath = inp.split("=")[-1]
        args.video_path = videoPath
        set_video(args)
        print("new video path set")
        actions["reset_loop"] = True
    if "setframes" in inp:
        frames_idx = inp.split("=")[-1]
        frames_idx = eval(frames_idx)
        set_video(args,video_frame_index=frames_idx)
        actions["reset_loop"] = True
    if inp=="exit":
        actions["exit_loop"] = True
        print("exiting...")
    return actions
    
def getVideoFrameCount(video_path):
    cv2_vr = cv2.VideoCapture(video_path)
    total_frames = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2_vr.release()
    return total_frames

def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames

def load_video(video_path,args, video_frame_index=[0,1,2,3,4,5,6,7]):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    # frame_idx = video_frame_index
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    # print("Selected frames index ", frame_idx)
    return spare_frames,frame_time,video_time,frame_idx

def init_main(args):

    global model, tokenizer,image_processor,context_len,cfg_pretrained

    # Initialize the model
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    else:
        pass

    # import pdb;pdb.set_trace()
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False


    


def get_model_output(prompt,file,batch_of_frames=None):
    sg_outputs = {
        # "objects_list": "",
        "triplets": ""
    }

    conv = conv_templates[args.conv_mode].copy()

    qs = prompt
    if args.add_time_instruction:
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(input_video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        qs = f'{time_instruciton}\n{qs}'
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
            
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # cur_prompt = question

    with torch.inference_mode():
        # model.update_prompt([[cur_prompt]])
        # import pdb;pdb.set_trace()
        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        if "mistral" not in cfg_pretrained._name_or_path.lower():
            output_ids = model.generate(inputs=input_ids, images=input_video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=2048, top_p=0.1,num_beams=1,use_cache=False, stopping_criteria=[stopping_criteria])
            # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        else:
            output_ids = model.generate(inputs=input_ids, images=input_video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=2048, top_p=0.1, num_beams=1, use_cache=False)
            # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)


        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # import pdb;pdb.set_trace()
        if "mistral" not in cfg_pretrained._name_or_path.lower():
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()
        sg_outputs["triplets"] = outputs
    
    return sg_outputs


def get_frame_by_frame_annot(frame_count, rels, sub_ob_jects_by_id):
    frames_dict = {}
    for i in range(frame_count+1):
        if i not in frames_dict.keys():
            frames_dict[i] = {
                "triplets": [],
                "bbox": []
            }
    
    assert len(frames_dict.keys())<=frame_count+1

    # print("total frames ", frame_count)
    # t_start1 = time.perf_counter()
    for rel in rels:
        sub, obj, predicate, annot_frames = rel

        for anno_frame_range in annot_frames:
            f_start, f_end = anno_frame_range
            # print("anno_frame_range ", anno_frame_range)
        

            for f_index in range(f_start,f_end):
                if f_index>frame_count:
                    continue

                subn = sub_ob_jects_by_id[sub]["category"]
                objn = sub_ob_jects_by_id[obj]["category"]

                # subj_data = sub_ob_jects_by_id[subject_tid]
                # obj_data = sub_ob_jects_by_id[object_tid]

                # current_frame_traj = trajectories[activity_range]
                # sub_bb, obj_bb = None, None
                # for curr_trj in current_frame_traj:
                # 	if curr_trj["tid"]==subject_tid:
                # 		sub_bb = curr_trj["bbox"]
                # 	if curr_trj["tid"]==object_tid:
                # 		obj_bb = curr_trj["bbox"]

                frames_dict[f_index]["triplets"].append([f"{subn}-{sub}", predicate, f"{objn}-{obj}"])
                # frames_dict[activity_range]["triplets"].append([f"{subj_data['category']}-{subj_data['tid']}", predicate, f"{obj_data['category']}-{obj_data['tid']}"])
                # frames_dict[activity_range]["bbox"].append([sub_bb, obj_bb])

    assert len(frames_dict.keys())<=frame_count+1

    return frames_dict

def get_frame_samples(total_frames,every_nth=4,frame_window_size=32,shift_step=3, total_shifts=100):
	frames_selected = []
	assert shift_step!=every_nth
	for i in range(0,total_shifts,shift_step):
		frames =[]
		for j in range(i, i+frame_window_size,every_nth):
			if j>total_frames:
				break
			frames.append(j)
		if len(frames)>=int(frame_window_size/every_nth):
			frames.sort()
			frames_selected.append(frames)
	return frames_selected

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", required=False)
    parser.add_argument("--output_dir", help="Directory to save the model results.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results", required=False)
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=8)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args)

    init_main(args)

    sg_eval_counts = {
        "total_obj_cnt" : 0,
        "total_pred_cnt" : 0,
        "total_sub_cnt" : 0,
        "correct_obj_pred_cnt" : 0,
        "correct_subj_pred_cnt" : 0,
        "correct_predicate_cnt" : 0,
        "gt_triplets_cnt": 0,
        "pred_triplets_cnt": 0,
        "correct_pred_triplets_cnt": 0,
        "total_predicted_triplets": 0
    }

    GtData = {
        "subjects": [],
        "objects": [],
        "predicates": []
    }

    PredData = {
        "subjects": [],
        "predicates": [],
        "objects": []
    }

    # TODO SET PATHS here for propts
    exec(open("/root/jbhoi/gits/Video-LLaVA/picklePrompt.py").read())
    defaultPrompt = "None"
    with open('/root/jbhoi/gits/Video-LLaVA/prompts.pkl', 'rb') as handle:
        b = pickle.load(handle)
        defaultPrompt = b["default"]

    # print(defaultPrompt)
    # exit()

    dataset_name = "vidor"
    version = args.output_dir

    splits = ["test"]
    # imagenet_vidvrd_root = "/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd"
    # imagenet_vidvrd_video_path = os.path.join(imagenet_vidvrd_root, "videos")
    # dataset = VidVRD(imagenet_vidvrd_root, imagenet_vidvrd_video_path, splits)

    # TODO set path
    data_root = '/root/jbhoi/gits/OpenPVSG/data'
    with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
        anno = json.load(f)
    print('Keys inside pvsg.json:', list(anno.keys()))
    print('Number of Object Classes:', len(anno['objects']['thing']))
    print('Number of Stuff Classes:', len(anno['objects']['stuff']))
    print('Number of Relation Classes:', len(anno['relations']))

    videos_root = os.path.join(data_root,dataset_name,'videos')
    inference_output_dir  = f"{data_root}/inference_outputs_onevision/{args.output_dir}" 
    inference_prog_output_dir  = f"{data_root}/inference_outputs_onevision/{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)

    llava_response_json = {}
    llava_raw_response_json = {}

    opvsg_predictes = anno['relations'] 
    opvsg_objects = anno['objects']['thing'] + anno['objects']['stuff']

    train_ids = anno["split"][dataset_name]["train"]
    val_ids = anno["split"][dataset_name]["val"]

    vidor_ids = train_ids + val_ids

  
    pbar = tqdm(total=len(val_ids))
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()


    # use only val ids for evaluation
    vidor_data = {data_dict['video_id']: data_dict for data_dict in anno['data'] if data_dict['video_id'] in val_ids}
    for id_idx, video_id in enumerate(val_ids):
        # if video_id!="0001_4164158586":
        #     continue
        video_data = vidor_data[video_id]

        total_frames = video_data["meta"]["num_frames"]
        vid_rels = video_data["relations"]

        summary = video_data["summary"]
        captions = video_data["captions"]

        vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in video_data["objects"]} 


        # frames_dict = get_frame_by_frame_annot(frame_count=total_frames,
        #                                        rels=vid_rels,
        #                                        sub_ob_jects_by_id=vid_objects_by_id)
        
        frameblocks = get_frame_samples(total_frames=total_frames,shift_step=5) # uniform sampling


        video_path = os.path.join(videos_root, video_id+".mp4")
        file = video_path if isinstance(video_path, list) else [video_path]
        args.video_path = video_path
        frame_idx = set_video(args=args, video_frame_index=frameblocks[0])
        # print(frame_idx)
        outputs_unclean = get_model_output(prompt=defaultPrompt,file=file)

        if video_id not in llava_raw_response_json:
            llava_raw_response_json[video_id] = {}

        llava_raw_response_json[video_id] = {
            "raw_response": outputs_unclean["triplets"],
            "frames": frame_idx,
            "GT": {
                "summary": summary,
                "captions": captions
            }
        }

        pbar.n +=1
        pbar.last_print_n = pbar.n
        pbar.refresh()

        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_pvsg_captions_raw_response_{version}.json"
            with open(outputfile, "w") as f:
                json.dump(str(llava_raw_response_json),f)
        except Exception as e:
            print(f"error saving file: {e}")