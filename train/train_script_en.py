import argparse
import json
import numpy as np
import pandas as pd
import os, sys

from LoadData import load_data, read_js
from preprocess import SummaryMatch, GreedyAlign

file_dir = os.path.dirname(os.path.dirname(__file__))

def get_args():

    # get dataset names
    ds_names = read_js(file_dir + "/DataStats/ds_name_list.json")

    # get command arguments
    parser = argparse.ArgumentParser(
        description="Get training parameters of datasets"
    )
    parser.add_argument(
        "-d", "--dataset", nargs=1, type=str, choices=ds_names, help="name of dataset"
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



if __name__ == '__main__':

    args = get_args()
    dataset = load_data(args.dataset[0], args.split[0], args.sample_propor)
    dataset_intermediate = SummaryMatch.extract_similar_summaries(dataset)
    dataset_intermediate = GreedyAlign.extract_greedy_summaries(dataset)