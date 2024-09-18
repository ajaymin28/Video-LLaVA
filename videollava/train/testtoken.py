import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, MAX_IMAGE_LENGTH, \
    MAX_VIDEO_LENGTH
from torch.utils.data import Dataset
from videollava.train.llava_trainer import LLaVATrainer

from videollava import conversation as conversation_lib
from videollava.model import *
from videollava.mm_utils import tokenizer_image_token

from PIL import Image
from videollava.utils import order_pick_k

from videollava.constants import SGSpecialTokens



def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # ================================================
    tokenizer_model_max_length: Optional[int] = None
    # ================================================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # ===================================================================
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    # ===================================================================

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    # ===================================================================
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: int = 8
    # ===================================================================


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:

            # ======================================================================================================
            if sentence['value'].startswith(DEFAULT_IMAGE_TOKEN) or sentence['value'].startswith(DEFAULT_VIDEO_TOKEN):  # run with multi-im, multi-vid, multi-im & multi-vid
                # <video><video><image><image>\nxxxxxxxxxxxxx  # must <video> first
                # <image>\nxxxxxxxxxxxxx -> <image>\nxxxxxxxxxxxxx
                # <video>\nxxxxxxxxxxxxx -> <video>\nxxxxxxxxxxxxx

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH).strip()
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH).strip()

            # a <video> is treated as `num_frames * <image>`
            replace_token, vid_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * data_args.num_frames
            # replace_token, vid_replace_token = DEFAULT_IMAGE_TOKEN, f"{SGSpecialTokens.VIDEO_FRAME_ID}{DEFAULT_IMAGE_TOKEN}" * data_args.num_frames   # JAAIMIN Changes
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN

            # <video><video><image><image>\nxxxxxxxxxxxxx -> `num_frames*<image>``num_frames*<image>`<image><image>\nxxxxxxxxxxxxx
            # <video>\nxxxxxxxxxxxxx -> `num_frames*<image>`\nxxxxxxxxxxxxx
            # print('before replace_token:', [sentence['value']])
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
            # print('after replace_token:', [sentence['value']])
            # ======================================================================================================

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    """
    sep=" ",
    sep2="</s>",
    """
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "  ## " ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
    #     return preprocess_plain(sources, tokenizer)
    # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
    #     return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # if conversation_lib.default_conversation.version == "mpt":
    #     return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    
    return None

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    """
    sep=" ",
    sep2="</s>",
    """
    # Mask targets
    Tok_mismatch = False
    sep = conv.sep + conv.roles[1] + ": "  ## " ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                Tok_mismatch =True

    return dict(
        input_ids=input_ids,
        labels=targets,
        Tok_mismatch=Tok_mismatch
    )

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

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


if __name__=="__main__":


    data_path = ["/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v18_3/videochatgpt_tune_.json"]
    list_data_dict = []
    for data in data_path:
        data = json.load(open(data, "r"))
        for i in data:
            i['id'] = len(list_data_dict)
            list_data_dict.append(i)


    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    ## Jaimin Changes
    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=dict(additional_special_tokens=['#frameseg', '#END_REL']),
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    
    class SGSpecialTokens:
        VIDEO_FRAME_ID = "#frameid"
        # SG_START = "#sg"
        SG_END = "#sgend"
        SG_SUBJECT = "#subject"
        SG_SUBJECT_ID = "#subid"
        SG_OBJECT = "#object"
        SG_OBJECT_ID = "#objid"
        SG_PREDICATE = "#sgpred"
        SG_BB_START = "#sgbb"
        SG_BB_END = "#sgbbend"
        SG_BB_X1Y1 = "#bbx1y1"
        SG_BB_X2Y2 = "#bbx2y2"
        # SG_BB_X1 = "#sgx1"
        # SG_BB_X2 = "#sgx2"
        # SG_BB_Y1 = "#sgy1"
        # SG_BB_Y2 = "#sgy2"

        @staticmethod
        def get_tokens():
            members = [attr for attr in dir(SGSpecialTokens) if not callable(getattr(SGSpecialTokens, attr)) and not attr.startswith("__")]
            tokens = []
            for mem in members:
                tokens.append(SGSpecialTokens.__getattribute__(SGSpecialTokens,mem))
            return tokens
        
    # special_tokens = SGSpecialTokens.get_tokens()

    # text = ": [cat-1:sitting on:table-9];[cat-1:walking on:table-9];#sgend"

    # text_list = ["[","cat","-","1",":","cat-1:","sitting","on","sitting on",":sitting on","[cat-1:sitting on:table-9]"]
    text_list = ['red_panda', 'lie_next_to', 'red_panda',':','_','#frameid','#frameid[red_panda:lie_next_to:red_panda]']

    # text_list = ["#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#frameid[lion-0:walk_front:lion-1];[lion-0:larger:lion-1];[lion-1:stand_behind:lion-0];#sgend"]


    # text = f"{SGSpecialTokens.VIDEO_FRAME_ID}"
    # text += f"[{SGSpecialTokens.SG_SUBJECT}'painting-{SGSpecialTokens.SG_SUBJECT_ID}9{SGSpecialTokens.SG_BB_START}-[{SGSpecialTokens.SG_BB_X1Y1}[1,2],{SGSpecialTokens.SG_BB_X2Y2}[3,4]]'"
    # text += f":{SGSpecialTokens.SG_OBJECT}'wall-{SGSpecialTokens.SG_OBJECT_ID}6{SGSpecialTokens.SG_BB_START}-[{SGSpecialTokens.SG_BB_X1Y1}[1,2],{SGSpecialTokens.SG_BB_X2Y2}[3,4]]'"
    # text += f":{SGSpecialTokens.SG_PREDICATE}'on'];"

    # text += f"[{SGSpecialTokens.SG_SUBJECT}'painting-{SGSpecialTokens.SG_SUBJECT_ID}9-[{SGSpecialTokens.SG_BB_X1Y1}[1,2],{SGSpecialTokens.SG_BB_X2Y2}[3,4]]'"
    # text += f":{SGSpecialTokens.SG_OBJECT}'wall-{SGSpecialTokens.SG_OBJECT_ID}6-[{SGSpecialTokens.SG_BB_X1Y1}[1,2],{SGSpecialTokens.SG_BB_X2Y2}[3,4]]'"
    # text += f":{SGSpecialTokens.SG_PREDICATE}'on'];{SGSpecialTokens.SG_END}"



    # num_added_toks = tokenizer.add_tokens(["#frameid"], special_tokens=True) ##This line is updated
    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=dict(additional_special_tokens=special_tokens),
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    tokenizer.do_lower_case = False


    for text in text_list:
        ids = tokenizer(text)["input_ids"]
        print(f"'{text}'=", ids)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print("decoded tokens ",decoded)

        if text==decoded:
            print("decoded is identical")
    
    exit()