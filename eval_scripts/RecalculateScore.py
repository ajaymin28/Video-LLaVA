import json
from utils.utilities import eval_tagging_scores, remove_ids
import numpy as np


with open("./inference_outputs_onevision/pvsg_v102/vidor_inference_val_pvsg_v102.json") as f:
    pred_data = eval(json.loads(f.read()))

overall_metric = {
    "subject": {"precision": [], "recall": []},
    "object": {"precision": [], "recall": []},
    "predicate": {"precision": [], "recall": []},
    "triplet": {"precision": [], "recall": []} 
}
for vid_id,vid_data in pred_data.items():


    block_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []}
    }

    for block_id, block_data in vid_data.items():
        pred_triplets = block_data["triplets"]
        frame_GT_triplets = block_data["GT_triplets"]
        frames = block_data["frames"]
        scene = block_data["scene"]
        st_progression = block_data["st_progression"]


        try:
            Block_GT_triplets_woids = remove_ids(frame_GT_triplets,version="v2_1", remove_indexes=True)
            Block_predicated_triplets_woids = remove_ids(pred_triplets,version="v2_1", remove_indexes=True)
        except Exception as e:
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

            
            
            for fm_key, fmdata in frame_metric.items():
                prec, rec, hit_scores = eval_tagging_scores(gt_relations=gt_all[fm_key],pred_relations=pred_all[fm_key],min_pred_num=1)
                frame_metric[fm_key]["precision"].append(prec)
                frame_metric[fm_key]["recall"].append(rec)
            # print(frame_GT_triplets,"<===>",frame_pred_triplets)
    
        for bm_key, bmdata in block_metric.items():
                block_metric[bm_key]["precision"].append(np.average(np.array(frame_metric[bm_key]["precision"], dtype=np.float32)))
                block_metric[bm_key]["recall"].append(np.average(np.array(frame_metric[bm_key]["recall"], dtype=np.float32)))

    for oam_key, oamdata in overall_metric.items():
        overall_metric[oam_key]["precision"].append(np.average(np.array(block_metric[oam_key]["precision"], dtype=np.float32)))
        overall_metric[oam_key]["recall"].append(np.average(np.array(block_metric[oam_key]["recall"], dtype=np.float32)))

sg_eval_counts = {}
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

print(sg_eval_counts)

