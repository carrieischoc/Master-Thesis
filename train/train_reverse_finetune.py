import os
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from utils import get_tokenized_dataset
from preprocess.utils import base_path, get_args


def get_seq2seq_args(model_name):

    path = os.path.join(
        base_path,
        "extend_models",
        model_name,
    )
    seq2seq_args = Seq2SeqTrainingArguments(
        output_dir=path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy="epoch",  # better estimation of model
        auto_find_batch_size=True,
        per_device_train_batch_size=8,  # defaults to 8, Larger batch sizes -> faster and better
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=8,
        # eval_delay=0.5,  # evaluate after training half epoch
        learning_rate=5e-5,
        num_train_epochs=5,  # increase the number gradually until the model stops improving
        lr_scheduler_type="linear",  # defaults to "linear"
        warmup_steps=975,  # 10% of the training steps
        weight_decay=0.01,
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

    return seq2seq_args


if __name__ == "__main__":

    # args.option[0] "extraction/lexrank_target_len", "extraction/lexrank_L"
    # "baselines/intermediate_greedy", "baselines/intermediate_top1",
    # "baselines/intermediate_top_extended_oracle_random",
    # "baselines/intermediate_top_extended_oracle_score"

    args = get_args()
    max_input_length = 192
    max_output_length = 160
    # get the training arguments
    # model_name: "wiki_lexrank_oracle_random"
    model_name = "news_wikis"
    seq2seq_args = get_seq2seq_args(model_name)

    # if args.dataset[0] == "GEM/xwikis_en":
    #     seq2seq_args.lr_scheduler_type="cosine"
    if args.dataset[0] == "billsum":        
        max_input_length = 832
        max_output_length = 448
        seq2seq_args.num_train_epochs = 10
        seq2seq_args.warmup_steps = 2354
        seq2seq_args.lr_scheduler_type="cosine"

    if args.dataset[0] == "cnn_dailymail":
        max_input_length = 192
        max_output_length = 128
        seq2seq_args.num_train_epochs = 7
        seq2seq_args.warmup_steps = 6000

    # Load the pre-trained model from checkpoint
    checkpoint = os.path.join(
        base_path,
        "cnn_dailymail",
        "models",
        "vanilla",
        "extraction",
        "lexrank_target_len",
        "192_128",
        "checkpoint-59059",
    )
    # don't use a Rust-based tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/mt5-base", model_max_length=512, use_fast=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    dataset_va = get_tokenized_dataset(
        tokenizer,
        base_path,
        args.dataset[0],
        "validation",
        options=args.option[0],
        drop_ratio=True,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
    )
    dataset_tr = get_tokenized_dataset(
        tokenizer,
        base_path,
        args.dataset[0],
        "train",
        options=args.option[0],
        drop_ratio=True,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
    )
  
    dataset_va = dataset_va.select(
            np.random.choice(dataset_va.num_rows, size=int(dataset_va.num_rows * 0.1), replace=False)
        )
    dataset_tr = dataset_tr.select(
            np.random.choice(dataset_tr.num_rows, size=int(dataset_tr.num_rows * 0.1), replace=False)
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=seq2seq_args,
        data_collator=data_collator,
        train_dataset=dataset_tr,
        eval_dataset=dataset_va,
        tokenizer=tokenizer,
        # optimizers=None,
        # preprocess_logits_for_metrics=None
    )

    # trainer.args.num_train_epochs += 5
    trainer.train()
