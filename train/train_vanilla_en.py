import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from utils import get_tokenized_dataset, load_from_path, get_single_tokenized_dataset
from preprocess.utils import base_path, get_args


def get_seq2seq_args(args, max_input_length: int, max_output_length: int):

    if args.debug == True:
        path = os.path.join(base_path, args.dataset[0], f"models/debug/vanilla")
        seq2seq_args = Seq2SeqTrainingArguments(
            output_dir=path,  # ./path/to/checkpoint
            # overwrite_output_dir=True,  # overwrite the content or continue training from the checkpoint
            do_train=True,  # performe training when the train() method is called
            do_eval=True,  # call the evaluate() method on the trainer object to start the evaluation process
            # do_predict=False, default to False
            evaluation_strategy="epoch",
            # eval_steps=50,
            auto_find_batch_size=True,  # avoid CUDA Out-of-Memory errors
            per_device_train_batch_size=4,  # The batch size per GPU/TPU core/CPU for training (default:8)
            per_device_eval_batch_size=4,  # for evluation
            gradient_accumulation_steps=8,  # Number of updates steps to accumulate before update
            eval_accumulation_steps=8,
            eval_delay=0.5,  # Number of epochs/steps to wait for before promising convergence
            learning_rate=5e-5,  # The initial learning rate, default
            num_train_epochs=350,
            lr_scheduler_type="constant",  # constant learning rate
            warmup_steps=50,
            logging_strategy="steps",
            logging_steps=50,
            # save_strategy="epoch",
            # save_steps=250,
            predict_with_generate=True,
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
            f"models/vanilla/{args.option[0]}/{str(max_input_length)}_{str(max_output_length)}",
        )
        seq2seq_args = Seq2SeqTrainingArguments(
            output_dir=path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy="epoch",  # better estimation of model
            auto_find_batch_size=True,
            per_device_train_batch_size=8,  # defaults to 8
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=8,
            # eval_delay=0.5,  # evaluate after training half epoch
            learning_rate=5e-5,
            num_train_epochs=5,  # increase the number gradually until the model stops improving
            lr_scheduler_type="linear",  # defaults to "linear"
            warmup_steps=9750,  # 10% of the training steps
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=250,
            save_strategy="epoch",  # Save is done at the end of each epoch.
            seed=2333,
            data_seed=128,
            # bf16=True,  # experimental feature, doesn't work on our Titan RTX GPUs!
            # bf16_full_eval=True,
            # fp16=True,
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
    # Load the pre-trained model and tokenizer
    # model_name = "google/mt5-base"
    model_name = "google/flan-t5-base"
    # don't use a Rust-based tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=512, use_fast=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if args.dataset[0] == "wikis_news_bills":
        max_input_length = 512
        max_output_length = 256
        seq2seq_args = get_seq2seq_args(
            args, max_input_length, f"{max_output_length}_t5"
        )
        # seq2seq_args.gradient_accumulation_steps = 8 # too large
        seq2seq_args.warmup_steps = 14172

        dataset_va = get_single_tokenized_dataset(
            tokenizer,
            base_path,
            "validation",
            options=args.option[0],
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
        dataset_tr = get_single_tokenized_dataset(
            tokenizer,
            base_path,
            "train",
            options=args.option[0],
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )

    else:

        if args.dataset[0] == "GEM/xwikis_en":
            # adapted from Dennis summaries library
            max_input_length = 176 # 264(L) 192(target_len)
            max_output_length = 136
            # get the training arguments
            seq2seq_args = get_seq2seq_args(
                args, max_input_length, f"{max_output_length}_t5"
            )

        # modify training arguments for different datasets
        if args.dataset[0] == "billsum":
            max_input_length = 840
            max_output_length = 416
            seq2seq_args = get_seq2seq_args(
                args, max_input_length, max_output_length
            )
            seq2seq_args.num_train_epochs = 10
            seq2seq_args.warmup_steps = 2354
            seq2seq_args.lr_scheduler_type = "cosine"

        if args.dataset[0] == "cnn_dailymail":
            max_input_length = 184
            max_output_length = 96
            seq2seq_args = get_seq2seq_args(
                args, max_input_length, max_output_length
            )
            seq2seq_args.num_train_epochs = 7
            seq2seq_args.warmup_steps = 6000

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

    if args.debug == True:
        # select 4 exs to debug and predict
        dataset_tr = dataset_tr.select(range(4))
        dataset_va = dataset_va.select(range(4))

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
    # trainer.train("checkpoint-96600")

    # test the predict
    if args.debug == True:
        path = os.path.join(base_path, args.dataset[0], f"models/debug/vanilla")
        trainer.save_model(path)
        dataset_tr = dataset_tr.remove_columns(
            ["decoder_input_ids", "decoder_attention_mask", "labels"]
        )
        predictions = trainer.predict(
            dataset_tr, max_length=128, num_beams=4, early_stopping=True
        )
        decoded_predictions = tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        ds_str = load_from_path(args.dataset[0], "train", base_path, "list_str_format")
        for i in range(4):
            print(f"({i}):")
            print("[predict]: ", decoded_predictions[i])
            print("[target]: ", ds_str[i]["target"])
