import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from preprocess.utils import base_path, get_args
from preprocess.split import check_combine_feature
from train.utils import load_from_path

"""
***************** No longer use it because it's too slow. *****************
"""

if __name__ == "__main__":

    args = get_args()
    tokenize_model_name = "google/mt5-base"
    input_max_length = 128
    tokenizer = AutoTokenizer.from_pretrained(
        tokenize_model_name, model_max_length=input_max_length, use_fast=False
    )

    def tokenize_func(example):
        return tokenizer(
            example["intermediate_summary"],
            padding="max_length",
            truncation=True,
            max_length=input_max_length,
            # return_tensors="pt",
        )

    try:
        dataset = load_from_path(
            args.dataset[0],
            "test",
            base_path,
            f"tokenized/lexrank_target_len_{input_max_length}",
        )
    except FileNotFoundError:
        dataset = load_from_path(
            args.dataset[0], "test", base_path, "extraction/lexrank_target_len"
        )
        ds_str = load_from_path(args.dataset[0], "test", base_path, "list_str_format")
        dataset = dataset.remove_columns("target")
        dataset = dataset.add_column(name="target", column=ds_str["target"])
        dataset = check_combine_feature(dataset, "intermediate_summary")
        dataset = dataset.map(tokenize_func, num_proc=4)
        dataset.save_to_disk(
            os.path.join(
                base_path,
                args.dataset[0],
                "test",
                f"tokenized/lexrank_target_len_{input_max_length}",
            )
        )
    # search all checkpoints
    path_to_checkpoint = os.path.join(
        base_path,
        args.dataset[0],
        f"models/vanilla/{args.option[0]}_{str(args.drop_ratio)}",
    )
    checkpoints = [
        os.path.join(path_to_checkpoint, f)
        for f in os.listdir(path_to_checkpoint)
        if "checkpoint" in f
    ]

    def pred_map(example):
        generated = model.generate(
            input_ids=torch.tensor([example["input_ids"]]),
            # attention_mask=torch.tensor(example["attention_mask"]),
            max_length=128,
            num_beams=4,
            early_stopping=True,  # could set to True
        )
        example["prediction"] = tokenizer.decode(generated[0], skip_special_tokens=True)
        return example

    # device = 'cuda:0'
    model = None
    for cp in checkpoints:
        if model is not None:
            del model
        model = AutoModelForSeq2SeqLM.from_pretrained(cp)
        dataset_pred = dataset.map(
            pred_map, remove_columns=["attention_mask", "input_ids"]
        )
        output_dir = os.path.join(cp, "results")
        dataset_pred.save_to_disk(output_dir)
