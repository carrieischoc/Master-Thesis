import os
import math
from typing import List
import numpy as np
from datasets import Dataset
from preprocess.utils import base_path, get_args


def compute_average_cpRatio(dataset_name: str, base_path: str):
    """
    Compute the average compression ratio of th e training dataset;
    Use it to compute the estimated target_len of test data.
    """
    path = os.path.join(base_path, dataset_name, "train", "list_list_format")
    dataset = Dataset.load_from_disk(path)
    dataset = dataset.map(
        lambda example: {"len": len(example["target"]) / len(example["source"])},
        num_proc=16,
    )
    cp_ratio = np.mean(dataset["len"])

    return cp_ratio


def generate_length(
    dataset_name: str,
    split: str,
    base_path: str,
    num_proc: int = 16,
):

    # Should first generate list_list format!
    path = os.path.join(base_path, dataset_name, split, "list_list_format")
    dataset = Dataset.load_from_disk(path)

    # if split == "test":
    #     cp_ratio = compute_average_cpRatio(dataset_name, base_path)

    def len_map(example):
        # example["source_len"] = len(example["source"])

        # if split == "test":
        #     example["target_len"] = math.ceil(example["source_len"] * cp_ratio)
        # else:
        #     example["target_len"] = len(example["target"])
        example["len_ratio"] = example["target_len"] / example["source_len"]
        if example["target_len"] == 1:
            example["L"] = 3
        else:
            example["L"] = int(
                example["target_len"] + round(5.0 / example["target_len"]) + 1
            )
        if example["L"] >= 0.5 * example["source_len"]:
            example["L"] = int(example["target_len"] + 1)

        return example

    dataset = dataset.map(len_map, num_proc=num_proc)
    dataset.save_to_disk(path)

    return dataset


if __name__ == "__main__":

    args = get_args()
    generate_length(args.dataset[0], args.split[0], base_path)
