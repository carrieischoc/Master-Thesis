import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    # DataCollatorForSeq2Seq,
)
from preprocess.utils import base_path, get_args
from preprocess.split import check_combine_feature
from train.utils import load_from_path


if __name__ == "__main__":

    args = get_args()
    tokenize_model_name = "google/mt5-base"
    input_max_length = 256
    tokenizer = AutoTokenizer.from_pretrained(
        tokenize_model_name, model_max_length=input_max_length, use_fast=False
    )

    def tokenize_func(example):
        inputs = tokenizer(
            example["intermediate_summary"],
            padding="max_length",
            truncation=True,
            max_length=input_max_length,
        )
        # initializing the decoder_input_ids with the padding token
        # decoder_input_ids = tokenizer('<pad>').input_ids
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            # "decoder_input_ids": decoder_input_ids,
        }

    try:
        dataset = load_from_path(
            args.dataset[0],
            "test",
            base_path,
            f"tokenized/lexrank_target_len_{str(input_max_length)}",
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
                f"tokenized/lexrank_target_len_{str(input_max_length)}",
            )
        )
    # search all checkpoints
    path_to_checkpoint = os.path.join(
        base_path,
        args.dataset[0],
        f"models/vanilla/{args.option[0]}_{str(args.drop_ratio)}", # modify _length before run
    )
    # checkpoints = [
    #     os.path.join(path_to_checkpoint, f)
    #     for f in os.listdir(path_to_checkpoint)
    #     if "checkpoint" in f
    # ]
    # the best checkpoint
    checkpoints = [os.path.join(path_to_checkpoint, "checkpoint-10")]

    model = None
    for cp in checkpoints:
        if model is not None:
            del model
        model = AutoModelForSeq2SeqLM.from_pretrained(cp)
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(cp, "predict"),
            per_device_eval_batch_size=8,
            predict_with_generate=True,
        )

        trainer = Seq2SeqTrainer(model=model, args=training_args, tokenizer=tokenizer)
        predictions = trainer.predict(
            dataset, max_length=128, num_beams=2, early_stopping=True
        )
        decoded_predictions = tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        dataset_pred = dataset.add_column("prediction", decoded_predictions)
        dataset_pred = dataset_pred.remove_columns(["attention_mask", "input_ids"])
        dataset_pred.save_to_disk(os.path.join(cp, "predict_2"))
