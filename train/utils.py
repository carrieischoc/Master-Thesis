import os
import json
from datasets import Dataset
from preprocess.filter import filter_out_indices, remove_noise
from preprocess.split import check_combine_feature


def load_from_path(dataset_name: str, split: str, base_path: str, name):
    path = os.path.join(base_path, dataset_name, split, name)
    dataset = Dataset.load_from_disk(path)

    return dataset


def read_js(filename):
    with open(filename) as f:
        js = json.load(f)

    return js


def write_js(filename, path, data):
    os.makedirs(path, exist_ok=True)
    f = open(os.path.join(path,filename), "w+")
    json.dump(data, f)


def get_tokenized_dataset(
    tokenizer,
    base_path: str,
    dataset_name: str,
    split: str,
    options: str,
    drop_ratio: bool = True,
    max_input_length: int = 128,
    max_output_length: int = 128,
):
    """
    drop_ratio: whether or not to drop samples with target/source ratio >= 0.5
    """
    try:
        dataset = load_from_path(
            dataset_name,
            split,
            base_path,
            f"tokenized/{options}/{str(max_input_length)}_{str(max_output_length)}",
        )
    except FileNotFoundError:
        try:
            dataset = load_from_path(
            dataset_name,
            split,
            base_path,
            f"tokenized/{options}/noise_removed",
        )
        except FileNotFoundError:
            ds_str = load_from_path(dataset_name, split, base_path, "list_str_format")
            dataset = load_from_path(dataset_name, split, base_path, name=options)
            dataset = dataset.remove_columns("target")
            dataset = dataset.add_column(name="target", column=ds_str["target"])
            dataset = check_combine_feature(dataset, "intermediate_summary")
            try:
                path = os.path.join(base_path, dataset_name, split, "tokenized")
                drop_indices = read_js(f"{path}/drop_indices_{str(drop_ratio)}")
            except FileNotFoundError:
                drop_indices = filter_out_indices(
                    dataset_name, split, base_path, drop_ratio=drop_ratio
                )
                write_js(f"drop_indices_{str(drop_ratio)}", path, drop_indices)

            dataset = dataset.filter(
                lambda example, indice: indice not in drop_indices, with_indices=True
            )

            # remove the noise
            df = dataset.to_pandas()
            df = remove_noise(df, ["intermediate_summary", "target"])
            dataset = Dataset.from_pandas(df)
            dataset.save_to_disk(os.path.join(
            base_path,
            dataset_name,
            split,
            f"tokenized/{options}/noise_removed"))


        def tokenize_function(example):
            # Tokenize inputs and targets
            inputs = tokenizer(
                example["intermediate_summary"],
                padding="max_length",
                truncation=True,
                max_length=max_input_length,
            )

            if split != "test":
                targets = tokenizer(
                    example["target"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_output_length,
                )

                # Return the input and target IDs and the attention masks
                return {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "decoder_input_ids": targets["input_ids"],
                    "decoder_attention_mask": targets["attention_mask"],
                    "labels": targets["input_ids"],
                }
            else:
                return {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    # "decoder_input_ids": decoder_input_ids,
                }
            # the following only outputs tokenized source
            # def tokenize_function(example):
            #     return tokenizer(
            #         text=example["intermediate_summary"],
            #         text_target=example["target"],
            #         padding="max_length",
            #         truncation=True,
            #         max_length=512,
            #     )
        if split != "test":
            dataset = dataset.map(
                tokenize_function,
                num_proc=16,
                remove_columns=["intermediate_summary", "target"],
            )
        else:
            dataset = dataset.map(
                tokenize_function,
                num_proc=16,
            )
            
        save_path = os.path.join(
            base_path,
            dataset_name,
            split,
            f"tokenized/{options}/{str(max_input_length)}_{str(max_output_length)}",
        )
        dataset.save_to_disk(save_path)

    return dataset
