import os
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

from utils.utilities import create_batch_frames, eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_v18
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from myutils.utilities import getRandomPrompt

from tqdm import tqdm

import argparse
import os
import copy

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
                file = video_processor(file, return_tensors='pt',frame_indices=frame_indices,total_frames=getVideoFrameCount(files[0]))['pixel_values'][0].to(model.device, dtype=torch.float16)
            
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
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit,
                                                                     device=args.device, cache_dir=args.cache_dir,use_custom_tokenizer=True)
    
    
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
    parser.add_argument("--model-base", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--file", nargs='+', type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default="v18_videoonly")
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
        "subject": [],
        "object": [],
        "predicate": []
    }

    PredData = {
        "subject": [],
        "predicate": [],
        "object": []
    }


    dataset_name = "vidvrd"
    version = args.output_dir
    # data_root = '/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data'
    # with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
    #     anno = json.load(f)
    # train_ids = anno["split"][dataset]["train"]
    # val_ids = anno["split"][dataset]["val"]
    # data = {data_dict['video_id']: data_dict for data_dict in anno['data'] if data_dict['video_id'] in val_ids}

    splits = ["test"]
    imagenet_vidvrd_root = "/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd"
    imagenet_vidvrd_video_path = os.path.join(imagenet_vidvrd_root, "videos")
    dataset = VidVRD(imagenet_vidvrd_root, imagenet_vidvrd_video_path, splits)

    sg_eval_counts["subsets"] = splits

    test_data_dir = os.path.join(imagenet_vidvrd_root, "test")
    test_anno_files = glob.glob(os.path.join(test_data_dir, "*.json")) 
    val_ids = []

    for test_annot in test_anno_files:
        filename = os.path.basename(test_annot)
        # filename = test_annot.split("/")[-1]
        filename = filename.split(".")[0]
        val_ids.append(filename)

    for val_id_idx, val_id in enumerate(val_ids):
        annot = dataset.get_anno(vid=val_id)
        frame_h, frame_w = annot["height"], annot["width"]
        frame_count = annot["frame_count"]
        video_id = annot["video_id"]
        video_fps = annot["fps"]
        sub_ob_jects = annot['subject/objects']
        sub_ob_jects_by_id = {obj["tid"]: obj  for obj in sub_ob_jects}
        rels = annot['relation_instances']
        trajectories = annot['trajectories']
        for rel in rels:
            begin_fid = rel['begin_fid']
            end_fid = rel['end_fid']
            subject_tid =rel['subject_tid']
            predicate = rel['predicate']
            if "_" in predicate:
                predicate = predicate.replace("_", " ")
            object_tid = rel['object_tid']
            for activity_range in range(begin_fid,end_fid):
                subj_data = sub_ob_jects_by_id[subject_tid]
                obj_data = sub_ob_jects_by_id[object_tid]
                if activity_range>frame_count:
                    continue
                if subj_data['category'] not in GtData["subject"]:
                    GtData["subject"].append(subj_data['category'])
                if predicate not in GtData["predicate"]:
                    GtData["predicate"].append(predicate)
                if obj_data['category'] not in GtData["object"]:
                    GtData["object"].append(obj_data['category'])



    pbar = tqdm(total=len(val_ids))
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

    for val_id_idx, val_id in enumerate(val_ids):

        annot = dataset.get_anno(vid=val_id)

        frame_h, frame_w = annot["height"], annot["width"]
        frame_count = annot["frame_count"]
        video_id = annot["video_id"]
        video_fps = annot["fps"]

        # video_path = os.path.join(videos_root, video_id+".mp4")

        sub_ob_jects = annot['subject/objects']
        sub_ob_jects_by_id = {obj["tid"]: obj  for obj in sub_ob_jects}

        rels = annot['relation_instances']
        trajectories = annot['trajectories']

        frames_dict = {}
        for i in range(frame_count):
            if i not in frames_dict.keys():
                frames_dict[i] = {
                    "triplets": [],
                    "bbox": [],
                    "pred_triplets": []
                }
        # print(frames_dict.keys())

        for rel in rels:
            begin_fid = rel['begin_fid']
            end_fid = rel['end_fid']
            subject_tid =rel['subject_tid']
            predicate = rel['predicate']
            object_tid = rel['object_tid']

            for activity_range in range(begin_fid,end_fid):
                subj_data = sub_ob_jects_by_id[subject_tid]
                obj_data = sub_ob_jects_by_id[object_tid]

                current_frame_traj = trajectories[activity_range]
                sub_bb, obj_bb = None, None
                for curr_trj in current_frame_traj:
                    if curr_trj["tid"]==subject_tid:
                        sub_bb = curr_trj["bbox"]
                    if curr_trj["tid"]==object_tid:
                        obj_bb = curr_trj["bbox"]

                
                if activity_range>frame_count:
                    continue
                
                frames_dict[activity_range]["triplets"].append([f"{subj_data['category']}-{subj_data['tid']}", predicate, f"{obj_data['category']}-{obj_data['tid']}"])
                frames_dict[activity_range]["bbox"].append([sub_bb, obj_bb])

        
        capture = None
        overall_annotations = []
        videos_root = os.path.join(imagenet_vidvrd_root, "videos")
        video_path = os.path.join(videos_root, video_id+".mp4")

        if os.path.exists(video_path):
            # print(video_path, "exists")
            # capture = cv2.VideoCapture(video_path)
            frame_indices = []	
            frame_counter = 0
            tripletes_for_current_block = ""
            tripletes_list_for_current_block = []
            current_block_triplet_data = {
                "subjects": [],
                "objects": [],
                "predicates": []
            }
            for frame_idx, frame_data in frames_dict.items():

                if frame_idx>frame_count:
                    continue
                
                # capture.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
                # ret, frame = capture.read()
                # if not ret:
                #     continue
                # h,w,c = frame.shape
                
                tripletes_for_current_block += f"{SGSpecialTokens.VIDEO_FRAME_ID}"

                max_triplets_to_add = 5
                added_triplets = []
                current_frame_triplets = []
                for index_to_draw, triplet in enumerate(frame_data["triplets"]):
                                
                    subj = triplet[0]
                    predicate = triplet[1]
                    if "_" in predicate:
                        predicate = predicate.replace("_", " ")
                    obj = triplet[2]

                    subj, subj_id = subj.split("-")
                    obj, obj_id = obj.split("-")

                    if subj not in current_block_triplet_data["subjects"]:
                        current_block_triplet_data["subjects"].append(subj)
                    
                    if obj not in current_block_triplet_data["objects"]:
                        current_block_triplet_data["objects"].append(obj)

                    if predicate not in current_block_triplet_data["predicates"]:
                        current_block_triplet_data["predicates"].append(predicate)
                    
                    construct_triplet = f"[{subj}-{subj_id}"
                    construct_triplet += f":{predicate}"
                    construct_triplet += f":{obj}-{obj_id}"
                    construct_triplet += f"];"
                    if construct_triplet not in added_triplets:
                            tripletes_for_current_block += construct_triplet
                            added_triplets.append(construct_triplet)
                            # current_frame_triplets.append([f"{subj}-{subj_id}", f"{predicate}", f"{obj}-{obj_id}"])
                            # v3_1 changes predicate last
                            current_frame_triplets.append([f"{subj}-{subj_id}", f"{obj}-{obj_id}",f"{predicate}"])
                
                
                if len(current_frame_triplets)>0:
                    frame_indices.append(frame_idx)
                    tripletes_list_for_current_block.append(current_frame_triplets)
                    # print("adding index", frame_idx)
                    frame_counter +=1

                    if len(frame_indices)>=8:
                        overall_annotations.append({
                            "frame_idxes": frame_indices,
                            "frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}",
                            "triplets": tripletes_list_for_current_block,
                            "current_block_triplet_data": copy.deepcopy(current_block_triplet_data) 

                        })

                        tripletes_for_current_block = ""
                        frame_counter = 0
                        frame_indices = []
                        tripletes_list_for_current_block = []
                        current_block_triplet_data = {
                            "subjects": [],
                            "objects": [],
                            "predicates": []
                        }

            
            if len(frame_indices)>0:
                frames_needs_to_be_added = 8 - len(frame_indices)  # for last batch match the 8 frames
                for i in range(frames_needs_to_be_added):
                    tripletes_for_current_block = f"{SGSpecialTokens.VIDEO_FRAME_ID}" + tripletes_for_current_block
                    frame_indices.insert(0, frame_indices[0]-1)

                    if len(frame_indices)>=8:
                        overall_annotations.append({
                            "frame_idxes": frame_indices,
                            "frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}",
                            "triplets": tripletes_list_for_current_block,
                            "current_block_triplet_data": current_block_triplet_data
                        })
                        
            # capture.release()


        # print("len of overall annot", len(overall_annotations))
        block_metric = {
            "subject": {"precision": [], "recall": []},
            "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
        for frame_block_index, overall_annot in enumerate(overall_annotations):

            # frame_idxes = overall_annot["frame_idxes"]
            # frames_GT_sgs = overall_annot["frames_sgs"]
            # frames_GT_triplets = overall_annot["triplets"]

            Block_GT_Triplets = overall_annot["triplets"]
            Block_frame_ids = overall_annot["frame_idxes"]

            current_block_triplet_data = copy.deepcopy(overall_annot["current_block_triplet_data"])

            final_subjects_list = get_varying_list(current_block_list=current_block_triplet_data["subjects"], 
                                            full_list=GtData["subject"], 
                                            fix_size=50) 

            final_objects_list = get_varying_list(current_block_list=current_block_triplet_data["objects"], 
                                            full_list=GtData["object"], 
                                            fix_size=50)

            final_predicates_list = get_varying_list(current_block_list=current_block_triplet_data["predicates"], 
                                            full_list=GtData["predicate"], 
                                            fix_size=50) # total 132 predicates in vidvrd
            

            TripletQ = getRandomPrompt(key='triplet_prompt', static=False)
            TripletQ = TripletQ.replace("{subjects}", ",".join(final_subjects_list))
            TripletQ = TripletQ.replace("{objects}", ",".join(final_objects_list))
            TripletQ = TripletQ.replace("{predicates}", ",".join(final_predicates_list))


            if val_id not in llava_response_json:
                llava_response_json[val_id] = {}
                llava_raw_response_json[val_id] = {}

            if frame_block_index not in llava_response_json[val_id].keys():
                llava_response_json[val_id][frame_block_index] = {}
                llava_raw_response_json[val_id][frame_block_index] = {}


            
            file = os.path.join(videos_root, video_id+".mp4")
            file = file if isinstance(file, list) else [file]
            outputs_unclean = get_model_output(file=file,batch_of_frames=Block_frame_ids, prompt=TripletQ)
            outputs = pre_clean_prediction_data_v18(outputs_unclean["triplets"])

            # print(outputs_unclean, Block_GT_Triplets)

            # print(file, outputs)

            llava_response_json[val_id][frame_block_index] = {
                # "objects_list": outputs["objects_list"],
                "triplets": outputs,
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets
            }

            llava_raw_response_json[val_id][frame_block_index] = {
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets,
                "raw": outputs_unclean["triplets"],
                "Prompt": TripletQ,
                "cleaned_output": outputs
            }


            Block_GT_triplets_woids = remove_ids(Block_GT_Triplets,version="v3_1")
            Block_predicated_triplets_woids = remove_ids(outputs,version="v3_1")

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
                    fgt_s, fgt_p, fgt_o = fgt  # v3_1 changes
                    gt_all["triplet"].append({"triplet": fgt, "score": 1.0})
                    gt_all["subject"].append({"triplet": fgt_s, "score": 1.0})
                    gt_all["predicate"].append({"triplet": fgt_p, "score": 1.0})
                    gt_all["object"].append({"triplet": fgt_o, "score": 1.0})

                for fpred in frame_pred_triplets:
                    fpred_s, fpred_p, fpred_o  = fpred # v3_1 changes
                    pred_all["triplet"].append({"triplet": fpred, "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

                    if fpred_s not in GtData["subject"]:
                        if fpred_s not in PredData["subject"]:
                            PredData["subject"].append(fpred_s)
                    if fpred_p not in GtData["predicate"]:
                        if fpred_p not in PredData["predicate"]:
                            PredData["predicate"].append(fpred_p)
                    if fpred_o not in GtData["object"]:
                        if fpred_o not in PredData["object"]:
                            PredData["object"].append(fpred_o)

                for fm_key, fmdata in frame_metric.items():
                    prec, rec, hit_scores = eval_tagging_scores(gt_relations=gt_all[fm_key],pred_relations=pred_all[fm_key],min_pred_num=1)
                    frame_metric[fm_key]["precision"].append(prec)
                    frame_metric[fm_key]["recall"].append(rec)

                
                if len(GT_tripdata)>0 and len(frame_pred_triplets)>0:
                    try:
                        results = calculate_accuracy_varying_lengths(gt_triplets=GT_tripdata,pred_triplets=frame_pred_triplets, remove_duplicates=False)
                    except Exception as e:
                        print(f"error calculating score for vid {val_id} block:{frame_block_index} fidx {fidx} actual_fidx: {Block_frame_ids[fidx]}")

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
                    # print(f"vid {val_id} block:{frame_block_index} fidx {fidx} actual_fidx:{Block_frame_ids[fidx]} lengt: {len(GT_tripdata)} lenpred: {frame_pred_triplets} outputs: {outputs}, unclean: {outputs_unclean}")


            for bm_key, bmdata in block_metric.items():
                block_metric[bm_key]["precision"].append(np.average(np.array(frame_metric[bm_key]["precision"], dtype=np.float32)))
                block_metric[bm_key]["recall"].append(np.average(np.array(frame_metric[bm_key]["recall"], dtype=np.float32)))
        
        
        for oam_key, oamdata in overall_metric.items():
            overall_metric[oam_key]["precision"].append(np.average(np.array(block_metric[oam_key]["precision"], dtype=np.float32)))
            overall_metric[oam_key]["recall"].append(np.average(np.array(block_metric[oam_key]["recall"], dtype=np.float32)))
                   

        pbar.n +=1
        pbar.last_print_n = pbar.n
        pbar.refresh()

        sg_eval_counts["VRDFormer_Logic"] = {}
        total_vid_ids = len(overall_metric["triplet"]["precision"])
        for metric_key, metric_values in overall_metric.items():
            if metric_key not in sg_eval_counts["VRDFormer_Logic"].keys():
                sg_eval_counts["VRDFormer_Logic"][metric_key] = {}
            overall_precision = np.average(np.array(overall_metric[metric_key]["precision"], dtype=np.float32))
            overall_recall = np.average(np.array(overall_metric[metric_key]["recall"], dtype=np.float32))
            sg_eval_counts["VRDFormer_Logic"][metric_key] = {
                "Precision@1": overall_precision,
                "Recall@1": overall_recall,
            }
        sg_eval_counts["VRDFormer_Logic"]["TotalVideos"] = total_vid_ids

        try:
            sg_eval_counts["dataset_meta"] ={
                "dataset_triplets_existing": GtData,
                "dataset_triplets_new": PredData
            }
        except Exception as e:
            pass


        inference_output_dir  = f"{imagenet_vidvrd_root}/inference_outputs/{args.output_dir}" 
        os.makedirs(inference_output_dir,exist_ok=True)
        
        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_{version}.json"
            with open(outputfile, "w") as f:
                json.dump(str(llava_response_json),f)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_raw_response_{version}.json"
            with open(outputfile, "w") as f:
                json.dump(str(llava_raw_response_json),f)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_{version}_eval_data.json"
            with open(outputfile, "w") as f:
                json.dump(str(sg_eval_counts),f)
        except Exception as e:
            print(f"error saving file: {e}")