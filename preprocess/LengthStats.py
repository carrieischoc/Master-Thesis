from typing import List
from datasets import Dataset


def len_map(example, feature_name: List[str]):

    for fe in feature_name:
        example[fe + "_len"] = len(example[fe])

    return example


def generate_length(
    dataset_name: str,
    split: str,
    base_path: str,
    feature_name: List[str],
    num_proc: int = 16,
):

    # Should first generate list_list format!
    dataset = Dataset.load_from_disk(
        base_path + dataset_name + "/" + split + "/" + "list_list_format"
    )

    map_dict = {"feature_name": feature_name}
    dataset = dataset.map(len_map, fn_kwargs=map_dict, num_proc=num_proc)
    dataset.save_to_disk(
        base_path + dataset_name + "/" + split + "/" + "list_list_format"
    )
