import os
import json
import glob
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
import numpy as np
import time
from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_onevision_v14_AG
from utils.utilities import get_substring_between
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from utils.utilities import get_AG_annotations_framewise, get_shuffled_list
from utils.utilities import unnormbb
from utils.utilities import AG_Objects,AG_relations

import argparse
import os
import copy
import argparse
import os
import copy
from tqdm import tqdm
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
import math
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoConfig

import cv2
import base64
import pickle
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import random
from typing import Dict
from utils.utilities import get_varying_list

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


def preProcessInputs(files, image_processor, video_processor, model, frame_indices=None):

    tensor = []
    special_token = []
    for file in files:
        if os.path.splitext(file)[-1].lower() in image_ext:
            file = image_processor.preprocess(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN]
        elif os.path.splitext(file)[-1].lower() in video_ext:
            if frame_indices is None:
                file = video_processor(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            else:
                """Jaimin Changes: """
                """Added fixed 99 frames to avoid reading video during inference"""
                file = video_processor(file, return_tensors='pt',frame_indices=frame_indices,total_frames=99)['pixel_values'][0].to(model.device, dtype=torch.float16)
            
            special_token += [DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames
            # special_token += [f"{SGSpecialTokens.VIDEO_FRAME_ID}{DEFAULT_IMAGE_TOKEN}"] * model.get_video_tower().config.num_frames
            # special_token += [f"{DEFAULT_IMAGE_TOKEN}"] * model.get_video_tower().config.num_frames
        else:
            raise ValueError(f'Support video of {video_ext} and image of {image_ext}, but found {os.path.splitext(file)[-1].lower()}')
        # print(file.shape)
        tensor.append(file)

    return tensor, special_token

def init_main(args):

    global roles, conv_mode, image_processor, video_processor, model, tokenizer

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    args.model_base = None
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit,
                                                                     device=args.device, cache_dir=args.cache_dir,use_custom_tokenizer=False)
    
    
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

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles


def get_model_output(file,batch_of_frames=None, prompt=None):
    sg_outputs = {
        # "objects_list": "",
        "triplets": ""
    }
    
    tensor, special_token = preProcessInputs(files=file,image_processor=image_processor,
                     video_processor=video_processor,model=model,frame_indices=batch_of_frames)
    

    conv = conv_templates[conv_mode].copy()

    # inp = question_1 = "list the objects present in the video"
    # inp = question_2 = f"Generate scene graph for the provided image, also chose the predicate from this list {','.join(predicates)}"
    if prompt is not None:
        inp = question_2 = prompt

    if getattr(model.config, "mm_use_im_start_end", False):
        inp = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + '\n' + inp
    else:
        inp = ''.join(special_token) + '\n' + inp

    # conv.append_message(conv.roles[0], inp)

    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()

    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=tensor,  # video as fake images
    #         do_sample=True if args.temperature > 0 else False,
    #         temperature=args.temperature,
    #         max_new_tokens=args.max_new_tokens,
    #         streamer=None,
    #         use_cache=True,
    #         stopping_criteria=[stopping_criteria])

    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    # conv.messages[-1][-1] = outputs

    # sg_outputs["objects_list"] = outputs


    # later messages
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,  # video as fake images
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=None,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs

    sg_outputs["triplets"] = outputs
    
    return sg_outputs


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="/dev/shm")
    parser.add_argument("--file", nargs='+', type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default="[zs][vicuna]_AG")
    parser.add_argument("--start_index", type=int, default=0,required=False)
    args = parser.parse_args()
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

    dataset_name = "ActionGnome"
    dataset_name_to_save = dataset_name

    # TODO SET PATHS here for propts
    exec(open("/home/jbhol/dso/gits/Video-LLaVA/picklePrompt.py").read())
    defaultPrompt = "None"
    with open('/home/jbhol/dso/gits/Video-LLaVA/prompts.pkl', 'rb') as handle:
        b = pickle.load(handle)
        defaultPrompt = b["version_14_AG_ZS_triplets"]

    print(defaultPrompt)
    version = args.output_dir


    continue_eval = False
    if args.start_index!=0:
        dataset_name_to_save += f"start_idx_{args.start_index}"
        # if args.prev_eval_data=="":
        #     raise Exception("Require prev_eval data path to continue previous eval")

    splits = ["test"]
    VIDEO_ROOT_PATH = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    AG_ANNOTATIONS_DIR = "/groups/sernam/datasets/ActionGenome/ActionGenome/annotations"
    CHUNK_N = 1000 # Q&A will be chunked into CHUNK_N parts
    AG_Annotations,dataset_meta,video_frame_data = get_AG_annotations_framewise(AG_ANNOTATIONS_DIR=AG_ANNOTATIONS_DIR, 
                                                                                subset=splits[0])



    inference_output_dir  = args.output_dir
    inference_prog_output_dir  = f"{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)

    sg_eval_counts["subsets"] = splits

    # AG_Prompt = getRandomPrompt(key='AG_Prompt', static=True)
    # AG_Prompt = AG_Prompt.replace("{objects_list}",  ",".join(get_shuffled_list(AG_Objects)) )
    # AG_Prompt = AG_Prompt.replace("{spatial_relations}", ",".join(get_shuffled_list(AG_relations["spatial"])))
    # AG_Prompt = AG_Prompt.replace("{contacting_relations}", ",".join(get_shuffled_list(AG_relations["contacting"])))
    # AG_Prompt = AG_Prompt.replace("{attention_relations}", ",".join(get_shuffled_list(AG_relations["attention"])))

    AG_relationsCombined = AG_relations["attention"]+AG_relations["spatial"]+AG_relations["contacting"]

    pbar = tqdm(total=len(AG_Annotations))
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()
    
    llava_response_json = {}
    llava_raw_response_json = {}
    frame_block = 0

    overall_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []} 
    }

    for val_id_idx,AG_Annotation in enumerate(AG_Annotations):

        if val_id_idx<args.start_index:
            ## To Continue unfinished job
            pbar.n = val_id_idx
            pbar.last_print_n = pbar.n
            pbar.refresh()
            continue

        video_id, video_annotations = AG_Annotation
        video_path = os.path.join(VIDEO_ROOT_PATH,video_id)
        if not os.path.exists(video_path):
            print(f"[ERROR] video doesnt exist at: {video_path}")
            raise FileNotFoundError()
        
        
        block_wise_GT_data = []
        frame_indices = []
        added_GT_triplets_frames = []
        added_GT_triplets_bb = []
        # frame_by_frame_sub_obj_pair_bb = {}
        frame_counter = 0
        frame_width_height = []
        for frame_id, frame_triplets,frame_triplets_bb in video_annotations:
            frame_int_idx = int(frame_id.split(".")[0])
            # print(frame_id, frame_int_idx)
            added_GT_triplets_frames.append(frame_triplets)
            added_GT_triplets_bb.append(list(frame_triplets_bb))
            frame_indices.append(frame_int_idx)

            # if f"frame-{frame_counter}" not in frame_by_frame_sub_obj_pair_bb.keys():
            #     frame_by_frame_sub_obj_pair_bb[f"frame-{frame_counter}"] = []

            # for bb_pairs in frame_triplets_bb:
            #     bbox1, bbox2, frame_size = bb_pairs
            #     frame_width_height = frame_size

            #     bbox1 = unnormbb(pred_box=bbox1,mask=None,normlize=True,img_h=frame_width_height[1],img_w=frame_width_height[0],decimal=2)
            #     bbox2 = unnormbb(pred_box=bbox2,mask=None,normlize=True,img_h=frame_width_height[1],img_w=frame_width_height[0],decimal=2)

            #     frame_by_frame_sub_obj_pair_bb[f"frame-{frame_counter}"].append([bbox1,bbox2])

            frame_counter +=1
            
            if len(frame_indices)>=8:
                AG_PromptAddedInput = copy.deepcopy(defaultPrompt)
                block_wise_GT_data.append({
                    "frame_idxes": frame_indices,
                    "triplets": added_GT_triplets_frames,
                    "triplets_bb": added_GT_triplets_bb,
                    "sgcls_prompt": copy.deepcopy(AG_PromptAddedInput)
                })

                frame_indices = []
                added_GT_triplets_frames = []
                added_GT_triplets_bb = []
                frame_counter = 0
                # frame_by_frame_sub_obj_pair_bb = {}


        # print(f"remaining frames: {len(frame_indices)}")
        if len(frame_indices)>0:
            AG_PromptAddedInput = copy.deepcopy(defaultPrompt)
            ## add remaning frames
            block_wise_GT_data.append({
                "frame_idxes": frame_indices,
                "triplets": added_GT_triplets_frames,
                "triplets_bb": added_GT_triplets_bb,
                "sgcls_prompt": copy.deepcopy(AG_PromptAddedInput)
            })
        block_metric = {
            "subject": {"precision": [], "recall": []},
            "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
        last_processed_time = None

        for frame_block_index, block_data in enumerate(block_wise_GT_data):

            if last_processed_time is None:
                last_processed_time = time.perf_counter()
            
            nowTime = time.perf_counter()
            print(f"Processing video: {video_id} Block {frame_block_index}/{len(block_wise_GT_data)} last processed in:{round((nowTime-last_processed_time),4)}")
            last_processed_time = nowTime
            
            Block_frame_ids = block_data["frame_idxes"]
            Block_GT_Triplets = block_data["triplets"]
            sgcls_prompt = block_data["sgcls_prompt"]

            if video_id not in llava_response_json:
                llava_response_json[video_id] = {}
                llava_raw_response_json[video_id] = {}

            if frame_block_index not in llava_response_json[video_id].keys():
                llava_response_json[video_id][frame_block_index] = {}
                llava_raw_response_json[video_id][frame_block_index] = {}
            
            
            file = os.path.join(VIDEO_ROOT_PATH, video_id)
            file = file if isinstance(file, list) else [file]
            outputs_unclean = get_model_output(file=file,batch_of_frames=Block_frame_ids, prompt=sgcls_prompt)

            outputs = pre_clean_prediction_data_onevision_v14_AG(outputs_unclean["triplets"])

            # import pdb
            # pdb.set_trace()

            llava_response_json[video_id][frame_block_index] = {
                # "objects_list": outputs["objects_list"],
                "triplets": outputs["triplets"],
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets
            }

            llava_raw_response_json[video_id][frame_block_index] = {
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets,
                "raw": outputs_unclean["triplets"],
                "Prompt": sgcls_prompt,
                "cleaned_output": outputs
            }


            try:
                Block_GT_triplets_woids = remove_ids(Block_GT_Triplets,version="v2_1",remove_indexes=True)
                Block_predicated_triplets_woids = remove_ids(outputs["triplets"],version="v2_1",remove_indexes=True)
            except Exception as e:
                print(f"error removing ids {e}")
                pass

            frame_metric = {
                "subject": {"precision": [], "recall": []},
                "object": {"precision": [], "recall": []},
                "predicate": {"precision": [], "recall": []},
                "triplet": {"precision": [], "recall": []}
            }
            for fidx, GT_tripdata in enumerate(Block_GT_triplets_woids):
                results = None

                frame_GT_triplets = GT_tripdata
                frame_pred_triplets = []

                try:frame_pred_triplets = Block_predicated_triplets_woids[fidx]
                except Exception as e:
                    pass

                gt_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},
                pred_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},

                gt_all = {"triplet": [],"subject": [],"object": [],"predicate": []}
                pred_all = {"triplet": [],"subject": [],"object": [],"predicate": []}

                for fgt in frame_GT_triplets:
                    fgt_s, fgt_p, fgt_o = fgt 
                    gt_all["triplet"].append({"triplet": fgt, "score": 1.0})
                    gt_all["subject"].append({"triplet": fgt_s, "score": 1.0})
                    gt_all["predicate"].append({"triplet": fgt_p, "score": 1.0})
                    gt_all["object"].append({"triplet": fgt_o, "score": 1.0})

                for fpred in frame_pred_triplets:
                    fpred_s, fpred_p, fpred_o  = fpred
                    pred_all["triplet"].append({"triplet": fpred, "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

                    if fpred_s not in AG_Objects:
                        if fpred_s not in PredData["subjects"]:
                            PredData["subjects"].append(fpred_s)
                    if fpred_p not in AG_relationsCombined:
                        if fpred_p not in PredData["predicates"]:
                            PredData["predicates"].append(fpred_p)
                    if fpred_o not in AG_Objects:
                        if fpred_o not in PredData["objects"]:
                            PredData["objects"].append(fpred_o)

                # import pdb
                # pdb.set_trace()

                for fm_key, fmdata in frame_metric.items():
                    """
                    Eval score for each frame
                    """
                    prec, rec, hit_scores = eval_tagging_scores(gt_relations=gt_all[fm_key],pred_relations=pred_all[fm_key],min_pred_num=1)
                    frame_metric[fm_key]["precision"].append(prec)
                    frame_metric[fm_key]["recall"].append(rec)

                
                if len(GT_tripdata)>0 and len(frame_pred_triplets)>0:
                    try:
                        results = calculate_accuracy_varying_lengths(gt_triplets=GT_tripdata,pred_triplets=frame_pred_triplets, remove_duplicates=False)
                    except Exception as e:
                        print(f"error calculating score for vid {video_id} block:{frame_block_index} fidx {fidx} actual_fidx: {Block_frame_ids[fidx]}")

                    if results is not None:
                        sg_eval_counts["correct_pred_triplets_cnt"] +=  results["correct_triplet_cnt"]
                        sg_eval_counts["correct_obj_pred_cnt"] += results["correct_object_cnt"]
                        sg_eval_counts["correct_subj_pred_cnt"] +=  results["correct_subject_cnt"]
                        sg_eval_counts["correct_predicate_cnt"] +=  results["correct_predicate_cnt"]
                        sg_eval_counts["gt_triplets_cnt"] +=  results["total_triplets"]
                        sg_eval_counts["total_predicted_triplets"] += results["total_predicted_triplets"]
                        sg_eval_counts["total_obj_cnt"] +=  results["total_objects"]
                        sg_eval_counts["total_sub_cnt"] +=  results["total_subjects"]
                        sg_eval_counts["total_pred_cnt"] +=  results["total_predicates"] 
                else:
                    pass
                    # print(f"vid {video_id} block:{frame_block_index} fidx {fidx} actual_fidx:{Block_frame_ids[fidx]} lengt: {len(GT_tripdata)} lenpred: {frame_pred_triplets} outputs: {outputs}, unclean: {outputs_unclean}")


            for bm_key, bmdata in block_metric.items():
                """
                    average eval score for each frame and appned it to block
                """
                if len(frame_metric[bm_key]["precision"])>0 and len(frame_metric[bm_key]["recall"])>0:
                    block_metric[bm_key]["precision"].append(np.average(np.array(frame_metric[bm_key]['precision'])))
                    block_metric[bm_key]["recall"].append(np.average(np.array(frame_metric[bm_key]['recall'])))
        
        
        for oam_key, oamdata in overall_metric.items():
            """
                    average eval score for each block and appned it to overall
            """
            if len(block_metric[oam_key]["precision"])>0 and len(block_metric[oam_key]["recall"])>0:
                overall_metric[oam_key]["precision"].append(round(float(np.average(np.array(block_metric[oam_key]['precision']))), 4))
                overall_metric[oam_key]["recall"].append(round(float(np.average(np.array(block_metric[oam_key]['recall']))), 4))

        try:
            with open(f"{inference_prog_output_dir}/{val_id_idx}_{len(AG_Annotations)}.txt", "w") as f:
                f.write(json.dumps(overall_metric, indent=4))
        except Exception as e:
            print(f"error saving file: {inference_prog_output_dir}/{val_id_idx}_{len(AG_Annotations)}.txt")
        
        pbar.n +=1
        pbar.last_print_n = pbar.n
        pbar.refresh()


        sg_eval_counts["VRDFormer_Logic"] = {}
        total_vid_ids = len(overall_metric["triplet"]["precision"])
        for metric_key, metric_values in overall_metric.items():
            if metric_key not in sg_eval_counts["VRDFormer_Logic"].keys():
                sg_eval_counts["VRDFormer_Logic"][metric_key] = {}
            
            if len(overall_metric[metric_key]["precision"])>0 and len(overall_metric[metric_key]["recall"])>0:
                overall_precision = np.average(np.array(overall_metric[metric_key]["precision"]))
                overall_recall = np.average(np.array(overall_metric[metric_key]["recall"]))
                sg_eval_counts["VRDFormer_Logic"][metric_key] = {
                    "Precision@1": round(float(overall_precision), 4),
                    "Recall@1": round(float(overall_recall), 4),
                }
        sg_eval_counts["VRDFormer_Logic"]["TotalVideos"] = total_vid_ids

        try:
            sg_eval_counts["dataset_meta"] ={
                "dataset_triplets_existing": GtData,
                "dataset_triplets_new": PredData
            }
        except Exception as e:
            pass

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_inference_val.json"
            # outputfile = f"{inference_output_dir}/results.json"
            with open(outputfile, "w") as f:
                json.dump(llava_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_inference_val_raw_response.json"
            # outputfile = f"{inference_output_dir}/results_raw_response.json"
            with open(outputfile, "w") as f:
                json.dump(llava_raw_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_results_eval_data.json"
            with open(outputfile, "w") as f:
                json.dump(sg_eval_counts,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

       