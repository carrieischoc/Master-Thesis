import random
from typing import List
import numpy as np
from operator import itemgetter
from datasets import Dataset
from preprocess.utils import base_path, get_args


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

    for old_name, new_name in old_features, new_features:
        dataset1 = dataset1.add_column(name=new_name, column=dataset2[old_name])

    return dataset1


def filter_length_oracle(
    dataset_name: str,
    split: str,
    base_path: str,
    m_mul: int,
    n_add: int,
    filter_method: str,
    num_proc: int = 16,
):
    """
    Only for the condition that n_add is not 0.
    It doesn't provide process for exceptions, so you should comfirm the correct concatenate version first.
    m_mul: multiplication coefficient of length.
    n_add: addition coefficient of length.
    """

    if n_add == 0:

        print("No need to filter!")
        return

    else:

        # load the concatenated version
        dataset = Dataset.load_from_disk(
            base_path
            + dataset_name
            + "/"
            + split
            + "/baselines/"
            + f"intermediate_top_concat"
        )

        map_dict = {"filter_method": filter_method, "m_mul": m_mul, "n_add": n_add}
        dataset = dataset.map(map_filter_method, fn_kwargs=map_dict, num_proc=num_proc)

        # save only selected features with target
        col_names = [
            f"intermediate_summary_m{str(m_mul)}_n{str(n_add)}",
            f"intermediate_summary_pos_m{str(m_mul)}_n{str(n_add)}"
            f"intermediate_summary_scores_m{str(m_mul)}_n{str(n_add)}",
        ]
        dataset = select_ds_column(dataset, col_names)
        dataset.save_to_disk(
            base_path
            + dataset_name
            + "/"
            + split
            + "/baselines/"
            + f"intermediate_top_m{str(m_mul)}_n{str(n_add)}"
        )


def map_filter_method(example, filter_method, m_mul, n_add):

    # Compute the indices of duplicates between length*m and length*(m+1)
    original_list = example[f"intermediate_summary{str(m_mul)}"]
    extend_list = example[f"intermediate_summary{str(m_mul+1)}"]
    extend_list_scores = example[f"intermediate_summary_scores{str(m_mul+1)}"]
    # indices in the extended list
    extend_list_len = len(extend_list)
    duplicates_indices = [extend_list.index(item) for item in original_list]
    diff_indices = list(set(range(extend_list_len)) - set(duplicates_indices))

    # select candidates to delete according to either score or random

    if filter_method == "oracle_score":
        sorted_indices = np.argsort(extend_list_scores).tolist()
        sorted_diff_indices = np.delete(sorted_indices, duplicates_indices)
        selected_indices = sorted_diff_indices[: extend_list_len - n_add]

    elif filter_method == "oracle_random":

        selected_indices = random.sample(diff_indices, extend_list_len - n_add)

    example[f"intermediate_summary_m{str(m_mul)}_n{str(n_add)}"] = np.delete(
        extend_list, selected_indices
    ).tolist()
    example[f"intermediate_summary_scores_m{str(m_mul)}_n{str(n_add)}"] = np.delete(
        extend_list_scores, selected_indices
    ).tolist()
    example[f"intermediate_summary_pos_m{str(m_mul)}_n{str(n_add)}"] = np.delete(
        example[f"intermediate_summary_pos{str(m_mul+1)}"], selected_indices
    ).tolist()

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
    dataset_concat = Dataset.load_from_disk(
        base_path + dataset_name + "/" + split + "/baselines/" + "intermediate_top1"
    )
    for fe in old_features:
        dataset_concat = dataset_concat.rename_column(fe, f"{fe}1")
    for i in range(2, max_multiplication + 1):
        dataset1 = dataset_concat
        dataset2 = Dataset.load_from_disk(
            base_path
            + dataset_name
            + "/"
            + split
            + "/baselines/"
            + f"intermediate_top{str(i)}"
        )
        new_features = [fe + str(i) for fe in old_features]
        dataset_concat = concatenate_datasets(
            dataset1, dataset2, old_features, new_features
        )

    dataset_concat.save_to_disk(
        base_path
        + dataset_name
        + "/"
        + split
        + "/baselines/"
        + f"intermediate_top_concat"
    )


if __name__ == "__main__":

    args = get_args()

    # concatenate top-ns dataset -> intermediate_summary_top(n)
    concat_ds_to_max(args.dataset[0], args.split[0], base_path, max_multiplication=3)
