import os
import random
import json
from typing import List
import numpy as np
from datasets import Dataset
from preprocess.utils import base_path, get_args

# from inspection import whitespace_token
from preprocess.split import check_combine_str


def select_ds_column(dataset, col: List[str]):
    col_all = dataset.column_names
    for c in col:
        col_all.remove(c)
    dataset = dataset.remove_columns(col_all)

    return dataset


def concatenate_datasets(
    dataset1, dataset2, old_features: List[str], new_features: List[str]
):
    """
    Add old features columns from dataset2 to dataset1 with new feature names.
    """

    for old_name, new_name in zip(old_features, new_features):
        dataset1 = dataset1.add_column(name=new_name, column=dataset2[old_name])

    return dataset1


def filter_length_oracle(
    dataset_name: str,
    split: str,
    base_path: str,
    filter_method: str,
    num_proc: int = 16,
):
    """
    L = n + round(3/n) + 1;
    if L >= 0.5*source_len, L = n + 1.
    """

    # load the concatenated version
    path_concat = os.path.join(
        base_path, dataset_name, split, "baselines/intermediate_top_concat"
    )
    dataset = Dataset.load_from_disk(path_concat)
    map_dict = {"filter_method": filter_method}

    dataset = dataset.map(map_filter_oracle, fn_kwargs=map_dict, num_proc=num_proc)

    # save only selected features with target
    col_names = [
        "intermediate_summary",
        "intermediate_summary_pos",
        "intermediate_summary_scores",
        "intermediate_summary_indices",
        "source",
        "target",
    ]
    dataset = select_ds_column(dataset, col_names)
    path_save = os.path.join(
        base_path,
        dataset_name,
        split,
        f"baselines/intermediate_top_extended_{filter_method}",
    )
    dataset.save_to_disk(path_save)


def map_filter_oracle(example, filter_method):

    # Ignore examples with the ratio (summary/reference sentence length) >=1.
    if example["len_ratio"] < 1:
        L = example["L"]
        target_len = example["target_len"]
        m_mul = L // target_len  # integer
        n_add = L - m_mul * target_len

        # n=1, L is either 2 or 3; or no need to compute extra candidates
        if target_len == 1 or n_add == 0:

            example["intermediate_summary"] = example[
                f"intermediate_summary{str(m_mul)}"
            ]
            example["intermediate_summary_scores"] = example[
                f"intermediate_summary_scores{str(m_mul)}"
            ]
            example["intermediate_summary_pos"] = example[
                f"intermediate_summary_pos{str(m_mul)}"
            ]
            example["intermediate_summary_indices"] = example[
                f"intermediate_summary_indices{str(m_mul)}"
            ]

        else:

            # Compute the indices of duplicates between length*m and length*(m+1)
            original_list = example[f"intermediate_summary{str(m_mul)}"]
            extend_list = example[f"intermediate_summary{str(m_mul+1)}"]
            extend_list_scores = example[f"intermediate_summary_scores{str(m_mul+1)}"]
            # indices in the extended list
            extend_list_len = len(extend_list)

            # cause errors if item not in origianl_list
            # duplicates_indices = [extend_list.index(item) for item in original_list]
            # set() is unordered and can change the order.
            duplicates = np.in1d(extend_list, original_list)
            duplicates_indices = list(np.where(duplicates == True)[0])
            diff_indices = list(np.where(duplicates == False)[0])

            # select candidates to delete according to either score or random
            # L = duplicates + keep; extend_len = (m+1)*target_len
            # extended_len = duplicates + keep + drop
            # np.delete is used to delete a certain slice, not a sublist!
            if filter_method == "oracle_score":
                sorted_indices = np.argsort(extend_list_scores)
                # sorted_diff_indices = np.delete(sorted_indices, duplicates_indices)
                drop_indices = sorted_indices[
                    ~np.in1d(sorted_indices, duplicates_indices)
                ][: extend_list_len - L]

            elif filter_method == "oracle_random":

                drop_indices = random.sample(
                    diff_indices, min(extend_list_len - L, len(diff_indices))
                )

            example["intermediate_summary"] = np.delete(
                extend_list, drop_indices
            ).tolist()
            example["intermediate_summary_scores"] = np.delete(
                extend_list_scores, drop_indices
            ).tolist()
            example["intermediate_summary_pos"] = np.delete(
                example[f"intermediate_summary_pos{str(m_mul+1)}"], drop_indices
            ).tolist()
            example["intermediate_summary_indices"] = np.delete(
                example[f"intermediate_summary_indices{str(m_mul+1)}"], drop_indices
            ).tolist()

    else:
        example["intermediate_summary"] = example["intermediate_summary1"]
        example["intermediate_summary_scores"] = example["intermediate_summary_scores1"]
        example["intermediate_summary_pos"] = example["intermediate_summary_pos1"]
        example["intermediate_summary_indices"] = example[
            "intermediate_summary_indices1"
        ]

    return example


def concat_ds_to_max(
    dataset_name: str, split: str, base_path: str, max_multiplication: int = 2
):
    """
    Concatenate top-ns dataset -> intermediate_summary_top(n)
    """
    max_multiplication = 3
    old_features = [
        "intermediate_summary",
        "intermediate_summary_pos",
        "intermediate_summary_scores",
    ]
    path_load = os.path.join(base_path, dataset_name, split, "list_list_format")
    dataset_concat = Dataset.load_from_disk(path_load)
    dataset_concat = select_ds_column(
        dataset_concat, ["L", "len_ratio", "target_len", "target"]
    )

    for i in range(1, max_multiplication + 1):
        dataset1 = dataset_concat
        path_load = os.path.join(
            base_path, dataset_name, split, f"baselines/intermediate_top{str(i)}"
        )
        dataset2 = Dataset.load_from_disk(path_load)
        new_features = [fe + str(i) for fe in old_features]
        dataset_concat = concatenate_datasets(
            dataset1, dataset2, old_features, new_features
        )

    path_save = os.path.join(
        base_path, dataset_name, split, "baselines/intermediate_top_concat"
    )
    dataset_concat.save_to_disk(path_save)


def filter_out_indices(
    dataset_name: str, split: str, base_path: str, drop_ratio: bool = True
):
    """
    Generate a list of indices to be filtered out with indices by certain criterias.
    """

    try:
        path1 = os.path.join(base_path, dataset_name, split, "str_str_format")
        dataset = Dataset.load_from_disk(path1)
    except:
        path2 = os.path.join(base_path, dataset_name, split, "list_str_format")
        dataset = Dataset.load_from_disk(path2)
        dataset = check_combine_str(dataset, ["source"])
        dataset.save_to_disk(path1)
    df = dataset.to_pandas()
    indices_to_drop = []

    # Filter out examples with the ratio (summary/reference token numbers) > 0.5.
    if drop_ratio == True:
        try:
            src = json.load(open(f"{base_path}/{dataset_name}/{split}_src_length.json"))
            tg = json.load(open(f"{base_path}/{dataset_name}/{split}_tg_length.json"))
        except:
            stats_src = whitespace_token(df["source"])
            stats_tg = whitespace_token(df["target"])
            src = stats_src.lens.tolist()
            tg = stats_tg.lens.tolist()
            f = open(f"{split}_src_length.json", "w+")
            json.dump(src, f)
            f = open(f"{split}_tg_length.json", "w+")
            json.dump(tg, f)
        ratio = np.array(tg) / np.array(src)
        ratio_indices = list(np.where(ratio > 0.5)[0])
        indices_to_drop += ratio_indices

    # indices_to_drop = list(df.index[df.loc[:, "len_ratio"] >= 1])

    # Filter out examples that are probably a list.
    indices_to_drop += list(df.index[df["source"].str.count(" - ") > 5])
    indices_to_drop += list(df.index[df["source"].str.count(" / ") > 5])

    indices_to_drop = list(set(indices_to_drop))
    indices_to_drop = [int(x) for x in indices_to_drop]

    return indices_to_drop  # set() is unordered and can change the order.


def remove_noise(df, features: List[str]):
    start_char = ["(", ";", ";", ",", ",", ";", ";", ",", ","]
    end_char = [")", "; ", ", ", ", ", "; ", ";", ",", ",", ";"]
    pattern1 = [f"\\{i}[^a-zA-Z0-9]*\\{j}" for i, j in zip(start_char, end_char)]
    pattern1 = "|".join(pattern1)
    pattern2 = [r"\(; ", r"\( ; ", r"\(, ", r"\( , "]
    pattern2 = "|".join(pattern2)
    for feature in features:
        df[feature] = df[feature].str.replace(pattern1, "", regex=True)
        df[feature] = df[feature].replace(pattern2, "(", regex=True)

    return df


if __name__ == "__main__":

    args = get_args()

    # concatenate top-ns dataset -> intermediate_summary_top(n)
    # concat_ds_to_max(args.dataset[0], args.split[0], base_path, max_multiplication=3)
    filter_length_oracle(
        args.dataset[0],
        args.split[0],
        base_path,
        filter_method=args.method[0],
    )
