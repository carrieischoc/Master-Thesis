import argparse
import json
import numpy as np
import pandas as pd
import os, sys
from datasets import Dataset, concatenate_datasets, load_dataset
from LoadData import load_data, read_js
from preprocess import SummaryMatch, GreedyAlign
from summaries.baselines import lexrank_st


def get_args():

    # get command arguments
    parser = argparse.ArgumentParser(
        description="Get training parameters of datasets"
    )
    parser.add_argument(
        "-d", "--dataset", nargs=1, type=str, help="name of dataset"
    )
    parser.add_argument("--split", nargs=1, type=str, help="split of dataset")
    parser.add_argument(
        "-p",
        "--sample_propor",
        nargs="?",
        type=float,
        const=1.0,
        default=1.0,
        help="proportion of samples",
    )
    

    args = parser.parse_args()

    return args

def select_ds_column(dataset, col):
    col_all = dataset.column_names
    col_all.remove(col)
    dataset = dataset.remove_columns(col_all)

    return dataset

if __name__ == '__main__':

    base_path = "/home/jli/working_dir/2022-jiahui-li-thesis/data/"

    args = get_args()
    dataset = load_data(args.dataset[0], args.split[0], args.sample_propor)

    try:
        dataset_intermediate_top3 = Dataset.load_from_disk(base_path+args.dataset[0]+"_intermediate_top3")
        
    except FileNotFoundError:
        dataset_intermediate_top3 = SummaryMatch.extract_similar_summaries(dataset)
        dataset_intermediate_top3.save_to_disk(base_path+args.dataset[0]+"_intermediate_top3")

    try:
        dataset_intermediate_greedy = Dataset.load_from_disk(base_path+args.dataset[0]+"_intermediate_greedy")
    except FileNotFoundError:
        dataset_src_ls = select_ds_column(dataset_intermediate_top3, "source") # list format of reference
        dataset_tg_str = select_ds_column(dataset, "target") # str format of summary
        dataset_ls = concatenate_datasets([dataset_src_ls, dataset_tg_str], axis=1)
        dataset_intermediate_greedy = GreedyAlign.extract_greedy_summaries(dataset_ls)
        dataset_intermediate_greedy.save_to_disk(base_path+args.dataset[0]+"_intermediate_greedy")

    # Extractive model Calling it on a list of pre-split sentences:
    # k = 10
    # lexrank_st(list_of_sentences, num_sentences=k)


