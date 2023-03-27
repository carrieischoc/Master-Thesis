import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    # DataCollatorForSeq2Seq,
)
from preprocess.utils import base_path, get_args
from train.utils import load_from_path


if __name__ == "__main__":

    args = get_args()
    tokenize_model_name = "google/mt5-base"
    input_max_length = 128
    tokenizer = AutoTokenizer.from_pretrained(
        tokenize_model_name, model_max_length=input_max_length, use_fast=False
    )

    def tokenize_func(example):
        inputs = tokenizer(
            example["intermediate_summary"],
            padding=True,
            truncation=True,
            max_length=input_max_length,
        )
        # initializing the decoder_input_ids with the padding token
        decoder_input_ids = tokenizer('<pad>').input_ids
        return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "decoder_input_ids": decoder_input_ids,
            }

    try:
        dataset = load_from_path(
            args.dataset[0],
            "test",
            base_path,
            f"tokenized/lexrank_target_len_{str(input_max_length)}_predict",
        )
    except FileNotFoundError:
        dataset = load_from_path(
            args.dataset[0], "test", base_path, "extraction/lexrank_target_len"
        )
        dataset = dataset.map(tokenize_func, num_proc=4)
        dataset.save_to_disk(
            os.path.join(
                base_path,
                args.dataset[0],
                "test",
                f"tokenized/lexrank_target_len_{str(input_max_length)}_predict",
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

    model = None

    # def decode(example):
    #     example["prediction"] = tokenizer.decode(
    #         example["prediction"], skip_special_tokens=True
    #     )
    #     return example

    for cp in checkpoints:
        if model is not None:
            del model
        model = AutoModelForSeq2SeqLM.from_pretrained(cp)
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(cp, "predict/prediction"),
            per_device_eval_batch_size=8,
            predict_with_generate=True,
        )

        trainer = Seq2SeqTrainer(model=model, args=training_args, tokenizer=tokenizer)
        predictions = trainer.predict(dataset, num_beams=4)

        decoded_predictions = []
        for pred in predictions.predictions:
            decoded_prediction = tokenizer.decode(pred, skip_special_tokens=True)
            decoded_predictions.append(decoded_prediction)
        dataset_pred = dataset.add_column("prediction", decoded_predictions)
        dataset_pred = dataset_pred.remove_columns(["attention_mask", "input_ids"])
        dataset_pred.save_to_disk(os.path.join(cp, "predict"))
