import os
import torch
from datasets import Dataset
from preprocess.utils import base_path, get_args
from preprocess.filter import select_ds_column
from BertSum.src.prepro import data_builder


def bertsum_extraction(args, dataset_name: str,
    split: str,
    base_path: str,
    extend_method: str):

    path = os.path.join(base_path, dataset_name, split, f"baselines/intermediate_{extend_method}")
    dataset = Dataset.load_from_disk(path)
    dataset = dataset.map(format_to_bert_map,num_proc=16)
    dataset_dict = []
    for ex in dataset:
        dataset_dict.append(ex["b_data_dict"])

    path = os.path.join(base_path, dataset_name, split, f"extraction/bertdata/{extend_method}")
    torch.save(dataset_dict, path)
    
def format_to_bert_map(example):
    bert = data_builder.BertData(args)
    b_data = bert.preprocess(example["source"], example["target"], example["intermediate_summary_indices"])
    indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
    b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
    example["b_data_dict"] = b_data_dict

    return example
    

if __name__ == "__main__":

    args = get_args()
    # method in choices of 'top1', 'top_extend_oracle_score', 'top_extend_oracle_random'
    bertsum_extraction(args,args.dataset[0], args.split[0], base_path, "top1")
