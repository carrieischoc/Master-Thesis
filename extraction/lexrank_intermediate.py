import os
from datasets import Dataset
from summaries.baselines import lexrank_st
from LoadData import load_data
from preprocess.split import check_split_sent
from preprocess.LengthStats import generate_length
from preprocess.utils import base_path, get_args
from preprocess.filter import select_ds_column


def lexrank_extraction(
    dataset_name: str,
    split: str,
    base_path: str,
    length_feature: str,
    sample_propor: float = 1.0,
):

    try:  # Calling it on a list of pre-split sentences
        path = os.path.join(base_path, dataset_name, split, "list_list_format")
        dataset = Dataset.load_from_disk(path)

    except FileNotFoundError:
        dataset = load_data(dataset_name, split, sample_propor)
        dataset = check_split_sent(dataset, ["source", "target"])
        dataset.save_to_disk(path)

    # check if list-list format contains "length" feature
    if "target_len" not in dataset.features:
        dataset = generate_length(dataset_name, split, base_path, ["target"])

    kwargs = {"length_feature": length_feature}
    dataset = dataset.map(lexrank_map, fn_kwargs=kwargs)
    # save only lexrank results and target
    col_names = ["target", "intermediate_summary"]
    dataset = select_ds_column(dataset, col_names)
    path = os.path.join(base_path, dataset_name, split, f"extraction/lexrank_{length_feature}")
    dataset.save_to_disk(path)


def lexrank_map(example, length_feature):
    """
    Use the model which show greatest performance on Sentence Embeddings, especially good for EN.
    Choose the gpu with id 1.
    """
    example["intermediate_summary"] = lexrank_st(
        example["source"],
        st_model="all-mpnet-base-v2",
        num_sentences=example[length_feature],
        device=1,
    )

    return example


if __name__ == "__main__":

    args = get_args()
    # length_feature: "L" or "target_len"
    lexrank_extraction(args.dataset[0], args.split[0], base_path, length_feature="L")
