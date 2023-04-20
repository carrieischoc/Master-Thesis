import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    # DataCollatorForSeq2Seq,
)
from preprocess.utils import base_path, get_args
from train.utils import get_tokenized_dataset


if __name__ == "__main__":

    args = get_args()
    tokenize_model_name = "google/mt5-base"
    max_input_length = 256
    max_output_length = 256
    tokenizer = AutoTokenizer.from_pretrained(
        tokenize_model_name, model_max_length=max_input_length, use_fast=False
    )
    test_option = "extraction/lexrank_target_len"
    dataset = get_tokenized_dataset(tokenizer,
        base_path,
        args.dataset[0],
        "test",
        options=test_option,
        max_input_length=max_input_length,
        max_output_length=max_output_length,)
    
    # search all checkpoints
    path_to_checkpoint = os.path.join(
        base_path,
        args.dataset[0],
        f"models/vanilla/{args.option[0]}/192_160_12", # modify _length before run
    )
    # checkpoints = [
    #     os.path.join(path_to_checkpoint, f)
    #     for f in os.listdir(path_to_checkpoint)
    #     if "checkpoint" in f
    # ]
    # the best checkpoint
    checkpoints = [os.path.join(path_to_checkpoint, "checkpoint-96600")]

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
            dataset, max_length=128, num_beams=4, early_stopping=True
        )
        decoded_predictions = tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        dataset_pred = dataset.add_column("prediction", decoded_predictions)
        dataset_pred = dataset_pred.remove_columns(["attention_mask", "input_ids"])
        dataset_pred.save_to_disk(os.path.join(cp, "predict"))
