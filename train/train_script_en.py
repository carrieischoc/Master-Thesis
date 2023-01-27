import argparse
import json
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, concatenate_datasets, load_dataset
from LoadData import load_data, read_js
from summaries.baselines import lexrank_st


def get_args():

    # get command arguments
    parser = argparse.ArgumentParser(description="Get training parameters of datasets")
    parser.add_argument("-d", "--dataset", nargs=1, type=str, help="name of dataset")
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


def get_seq2seq_args(debug: bool = False) -> Seq2SeqTrainingArguments:
    if debug:
        args = Seq2SeqTrainingArguments(
            output_dir="./xwiki_en_seq2seq",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="steps",  # Evaluation is done (and logged) every eval_steps
            eval_steps=50,
            auto_find_batch_size=True,
            # per_device_train_batch_size=4, # The batch size per GPU/TPU core/CPU for training (default:8)
            per_device_eval_batch_size=4,  # for evluation
            gradient_accumulation_steps=4,  # Number of updates steps to accumulate the gradients
            # eval_accumulation_steps=20, If unset, accumulated all on GPU/TPU before -> CPU (faster but more memory)
            # eval_delay=0.5, Number of epochs or steps to wait for before the first evaluation
            learning_rate=5e-5, # The initial learning rate
            num_train_epochs=1000,
            lr_scheduler_type="constant", # constant learning rate
            warmup_steps=50,  # roughly equivalent to 1/6 of the first epoch
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="no",
            seed=768,
            data_seed=512,
            # Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training
            # bf16=True,  # experimental feature, doesn't work on our Titan RTX GPUs!
            # bf16_full_eval=True,
            optim="adamw_torch", # optimizer
            gradient_checkpointing=False,  # If True, save memory at the expense of slower backward pass
        )
    else:
        args = Seq2SeqTrainingArguments(
            output_dir="./xwiki_en_seq2seq",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="epoch",  # Evaluation is done at the end of each epoch
            auto_find_batch_size=True,
            # per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            # eval_accumulation_steps=20,
            # eval_delay=0.5,
            learning_rate=5e-5,
            num_train_epochs=7,
            # lr_scheduler_type="linear", # defaults to "linear"
            warmup_steps=5000,  # roughly equivalent to 1/6 of the first epoch
            logging_strategy="steps",
            logging_steps=250,
            save_strategy="epoch", # Save is done at the end of each epoch.
            seed=768,
            data_seed=512,
            # bf16=True,  # experimental feature, doesn't work on our Titan RTX GPUs!
            # bf16_full_eval=True,
            run_name="GEM/xwikis English Summarization",
            optim="adamw_torch", # optimizer
            gradient_checkpointing=False,  
        )

    return args


def select_ds_column(dataset, col):
    col_all = dataset.column_names
    col_all.remove(col)
    dataset = dataset.remove_columns(col_all)

    return dataset


if __name__ == "__main__":

    base_path = "/home/jli/working_dir/2022-jiahui-li-thesis/data/"

    # adapted from Dennis summaries library
    # debug = True

    # if debug:
    #     model_name = "google/mt5-small"
    #     max_length = 256
    #     summary_max_length = 128
    # else:
    #     model_name = "google/mt5-base"
    #     max_length = 768
    #     summary_max_length = 512

    # tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length, use_fast=False)
    # # Enable this for T5-based models
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # data_files = {
    #     "train": "xwiki_en_summarization.jsonl",
    #     "validation": "xwiki_en_validation.jsonl",
    # }
    # if debug:
    #     data_files["train"] = "xwiki_en_validation.jsonl"

    # if debug:
    #     dataset["train"] = dataset["train"].select(range(5))
    #     dataset["validation"] = dataset["validation"].select(range(5))

    # tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # args = get_seq2seq_args(debug=debug)

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=args,
    #     data_collator=data_collator,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["validation"],
    #     tokenizer=tokenizer,
    #     # optimizers=None,
    #     # preprocess_logits_for_metrics=None
    # )

    # trainer.train()


    # Extractive model Calling it on a list of pre-split sentences:
    # k = 10
    # lexrank_st(list_of_sentences, num_sentences=k)
