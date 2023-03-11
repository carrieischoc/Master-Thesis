import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict
from utils import load_from_path
from preprocess.utils import base_path, get_args
from preprocess.filter import filter_out_indices


def filter_out_dataset(dataset_name: str, split: str, base_path: str, drop_ratio: bool):
    dataset = load_from_path(dataset_name, split,base_path, f"extraction/lexrank_{args.method[0]}")
    ds_str = load_from_path(dataset_name, split,base_path,"list_str_format")
    dataset = dataset.remove_columns("target")
    dataset = dataset.add_column(name="target", column=ds_str["target"])
    drop_indices = filter_out_indices(
        dataset_name, split, base_path, drop_ratio=drop_ratio
    )
    
    dataset = dataset.filter(
        lambda example, indice: indice not in drop_indices, with_indices=True
    )
    return dataset


def get_seq2seq_args(args):

    dataset_tr = filter_out_dataset(
        args.dataset[0], "train", base_path, drop_ratio=args.drop_ratio
    )
    dataset_va = filter_out_dataset(
        args.dataset[0], "validation", base_path, drop_ratio=args.drop_ratio
    )

    if args.debug:
        # select the first 5% data to debug
        dataset_tr = dataset_tr.select(range(int(dataset_tr.num_rows * 0.05)))
        dataset_va = dataset_va.select(range(int(dataset_va.num_rows * 0.05)))
        path = os.path.join(
            base_path, args.dataset[0], f"models/debug/vanilla_{args.method[0]}"
        )
        seq2seq_args = Seq2SeqTrainingArguments(
            output_dir=path,  # ./path/to/checkpoint
            overwrite_output_dir=True,  # overwrite the content or continue training from the checkpoint
            do_train=True,  # performe training when the train() method is called
            do_eval=True,  # call the evaluate() method on the trainer object to start the evaluation process
            # do_predict=False, default to False
            evaluation_strategy="steps",  # faster
            eval_steps=50,
            auto_find_batch_size=True,  # avoid CUDA Out-of-Memory errors
            per_device_train_batch_size=4,  # The batch size per GPU/TPU core/CPU for training (default:8)
            per_device_eval_batch_size=4,  # for evluation
            gradient_accumulation_steps=8,  # Number of updates steps to accumulate before update
            eval_accumulation_steps=16,
            # eval_delay=0.5, # Number of epochs/steps to wait for before promising convergence
            learning_rate=5e-5,  # The initial learning rate, default
            num_train_epochs=1000,
            lr_scheduler_type="constant",  # constant learning rate
            warmup_steps=50,  # roughly equivalent to 1/6 of the first epoch
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="no",
            seed=1123,  # produces the same results
            data_seed=512,  # different from seed, try different values to help avoid overfitting
            # Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training
            # bf16=True,  # experimental feature, doesn't work on our Titan RTX GPUs!
            # bf16_full_eval=True,
            optim="adamw_torch",  # optimizer
            gradient_checkpointing=False,  # If True, save memory at the expense of slower backward pass
        )
    else:
        path = os.path.join(
            base_path,
            args.dataset[0],
            f"models/vanilla_{args.method[0]}_{str(args.drop_ratio)}",
        )
        seq2seq_args = Seq2SeqTrainingArguments(
            output_dir=path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="epoch",  # better estimation of model
            auto_find_batch_size=True,
            # per_device_train_batch_size=8, # defaults to 8, Larger batch sizes -> faster and better
            # per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,  # a large value needs more memory
            eval_accumulation_steps=16,
            eval_delay=2,  # try different values to earn more meaningful representations before the evaluation
            learning_rate=5e-5,
            num_train_epochs=5,  # increase the number gradually until the model stops improving
            # lr_scheduler_type="linear", # defaults to "linear"
            warmup_steps=5000,  # roughly equivalent to 1/6 of the first epoch
            logging_strategy="steps",
            logging_steps=250,
            save_strategy="epoch",  # Save is done at the end of each epoch.
            seed=2333,
            data_seed=128,
            # bf16=True,  # experimental feature, doesn't work on our Titan RTX GPUs!
            # bf16_full_eval=True,
            run_name="GEM/xwikis English Summarization",
            optim="adamw_torch",  # optimizer
            gradient_checkpointing=False,
        )

    dataset = DatasetDict({"train": dataset_tr, "validation": dataset_va})
    return dataset, seq2seq_args


if __name__ == "__main__":

    args = get_args()
    dataset, seq2seq_args = get_seq2seq_args(args)

    # adapted from Dennis summaries library
    # Load the pre-trained model and tokenizer
    model_name = "google/mt5-base"
    max_input_length = 512
    # don't use a Rust-based tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=max_input_length, use_fast=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize_function(example):
        # Tokenize the source and target sequences
       return tokenizer(
        text=example["intermediate_summary"],
        text_target=example["target"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # get the training arguments and dataset
    dataset, seq2seq_args = get_seq2seq_args(args)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=seq2seq_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        # optimizers=None,
        # preprocess_logits_for_metrics=None
    )

    trainer.train()

    # Evaluate the model
    # trainer.evaluate()
