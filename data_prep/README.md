## Data prep for pvsg

Follow pvsg envrionment setup and download data from [PVSG](https://github.com/LilyDaytoy/OpenPVSG)

Alternatively data can be downloaded from huggingface (wget is easier from HF)

- https://huggingface.co/datasets/Jingkang/PVSG
- https://huggingface.co/datasets/shangxd/imagenet-vidvrd


## Preparing Q&A annotations for video-llava

### PVSG

```
python /home/jbhol/dso/gits/VideoLLAVAGit/Video-LLaVA/data_prep/OPVSG_VIdeoLLAVA_Annot_chatgpt_v17.py --data_root=/home/jbhol/dso/gits/OpenPVSG/data/ --output_dir=out_dir --dataset=vidor
```

### VidVRD

```

[0,4,8,12,16,20,24,28] then shift by n=5
[5,9,13,18,22,26,30,34] then again shift by n=5
[10,14 .....]

python /home/jbhol/dso/gits/VideoLLAVAGit/Video-LLaVA/data_prep/prepare_video_llava_v7_newsampling.py

```


```
With time blocks triplet_[Frame-start,Frame-end]

[0,4,8,12,16,20,24,28] then shift by n=5
[5,9,13,18,22,26,30,34] then again shift by n=5
[10,14 .....]

python /home/jbhol/dso/gits/VideoLLAVAGit/Video-LLaVA/data_prep/prepare_video_llava_v7_newsampling_with_time.py

```
