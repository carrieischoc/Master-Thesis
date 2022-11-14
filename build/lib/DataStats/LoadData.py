import argparse
import json
import numpy as np
import pandas as pd
import os, sys
from typing import List
from datasets import load_dataset, Dataset

file_dir = os.path.dirname(os.path.dirname(__file__))

def read_js(filename):
    with open(filename) as f:
        js = json.load(f)

    return js


def get_args():

    # get dataset names
    ds_names = read_js("ds_name_list.json")

    # get command arguments
    parser = argparse.ArgumentParser(
        description="Get different information of different datasets"
    )
    parser.add_argument(
        "-d", "--dataset", nargs=1, type=str, choices=ds_names, help="name of dataset"
    )
    parser.add_argument("--split", nargs=1, type=str, help="split")
    parser.add_argument(
        "-t",
        "--token_method",
        choices=["whitespace", "spacy"],
        nargs=1,
        type=str,
        help="tokenization method: whitespace, spacy",
    )
    parser.add_argument(
        "--stats",
        nargs=1,
        type=str,
        choices=["similarity", "length", "samples_len"],
        help="type of stats: similarity, length, representative samples according to length",
    )
    # parser.add_argument(
    #     "--sf", nargs=1, type=str, help="if save the figure: 1, otherwise: 0"
    # )
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


def rename_datasets(dataset):

    source_names = read_js(file_dir + "/DataStats/source_names.json")
    target_names = read_js(file_dir + "/DataStats/target_names.json")

    flag_src = False  # check if source is renamed
    flag_tg = False  # check if target is renamed
    for col_name in dataset.column_names:
        if col_name in source_names:
            flag_src = True
            if col_name == "source":
                pass
            else:
                dataset = dataset.rename_column(col_name, "source")
        elif col_name in target_names:
            flag_tg = True
            if col_name == "target":
                pass
            else:
                dataset = dataset.rename_column(col_name, "target")
    if flag_src == False:
        raise ValueError(
            "Invalid name of source! Please add names in source_names.json."
        )
    if flag_tg == False:
        try:
            if dataset.features["source"].feature._type == "Value":
                raise NameError(
                    "Invalid name of target! Please add names in target_names.json."
                )
        # check if source and target are encapsulated
        except AttributeError:
            if len(dataset.features["source"].feature) > 1:
                dataset = pd.DataFrame(dataset["source"])
                dataset = Dataset.from_pandas(dataset)
                dataset = rename_datasets(dataset)

    return dataset


def remove_empty(dataset):
    dataset = dataset.filter(lambda x: x["source"] != [])
    dataset = dataset.filter(lambda x: x["source"] != "")
    dataset = dataset.filter(lambda x: x["target"] != [])
    dataset = dataset.filter(lambda x: x["target"] != "")

    return dataset


def load_data(dataset_name: str, split: str = "train", data_proportion: float = 1.0):

    
    version_dic = read_js(file_dir + "/DataStats/ds_version_dict.json")
    
    # reconstructing the data as a dictionary
    version = version_dic.get(dataset_name)

    if dataset_name == "scitldr_A" or dataset_name == "scitldr_F":
        dataset_name = "scitldr"
    if "GEM/xwikis" in dataset_name:
        dataset_name = "GEM/xwikis"
    
    dataset = load_dataset(dataset_name, version, split=split)
    dataset = dataset.flatten()

    # rename dataset columns as source and target
    dataset = rename_datasets(dataset)

    # remove empty examples in dataset
    dataset = remove_empty(dataset)

    # use a certain proportion of samples
    if data_proportion != 1.0:
        n = int(dataset.num_rows * data_proportion)
        dataset = dataset.select(
            np.random.choice(dataset.num_rows, size=n, replace=False)
        )

    return dataset
