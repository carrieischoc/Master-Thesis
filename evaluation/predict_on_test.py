import os
import re
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    # DataCollatorForSeq2Seq,
)
from preprocess.utils import base_path, get_args
from train.utils import get_tokenized_dataset
from utils import get_domain_specific_tokenized_testdata

if __name__ == "__main__":

    args = get_args()
    # tokenize_model_name = "google/mt5-base"
    tokenize_model_name = "google/flan-t5-base"
    # max_output_length will not be used
    max_input_length = 192
    if args.dataset[0] == "billsum":
        max_input_length = 680
    if args.dataset[0] == "cnn_dailymail":
        max_input_length = 256

    tokenizer = AutoTokenizer.from_pretrained(
        tokenize_model_name, model_max_length=512, use_fast=False
    )
    test_option = "extraction/lexrank_target_len"
    # dataset = get_tokenized_dataset(
    #     tokenizer,
    #     base_path,
    #     args.dataset[0],
    #     "test",
    #     options=test_option,
    #     max_input_length=max_input_length,
    # )

    # domain: "Wikipedia Text: ", "News Text: ", "Bills Text: "
    dataset = get_domain_specific_tokenized_testdata(
        tokenizer,
        base_path,
        args.dataset[0],
        options=test_option,
        domain="wikis",
        max_input_length=max_input_length,
    )
    path_to_checkpoint = os.path.join(
        base_path,
        "wikis_news_bills",
        "models",
        "vanilla",
        "extraction",
        "lexrank_target_len",
        "512_256",
    )

    # path_to_checkpoint = os.path.join(
    #     base_path,
    #     # args.dataset[0],
    #     "GEM/xwikis_en",
    #     # "billsum",
    #     # "cnn_dailymail",
    #     "models",
    #     "vanilla",
    #     "extraction",
    #     "lexrank_target_len",
    #     # "192_128",
    #     "192_160_10"
    #     # "832_448_cos",  # modify _length before run
    # )
    # path_to_checkpoint = os.path.join(base_path, "extend_models", "news_wikis")
    checkpoints = [f for f in os.listdir(path_to_checkpoint) if "checkpoint" in f]
    # the best checkpoint (largest)
    numbers = []
    for cp in checkpoints:
        numbers.append(int(re.findall("\d+", cp)[0]))
    best_cp = checkpoints[numbers.index(max(numbers))]
    checkpoints = os.path.join(path_to_checkpoint, best_cp)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoints)
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(checkpoints, "predict_wiki-news"),
        per_device_eval_batch_size=8,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(model=model, args=training_args, tokenizer=tokenizer)
    # predictions = trainer.predict(
    #     dataset,
    #     max_length=256,
    #     num_beams=1,
    #     early_stopping=True,
    #     top_k=30,
    #     top_p=0.95,
    #     temperature=0.9,
    #     do_sample=True,
    # )
    predictions = trainer.predict(
        dataset,
        max_length=256,
        num_beams=4,  # Adjust the number of beams accordingly
        # num_beam_groups=4,  # Divide beams into groups to encourage diversity
        # diversity_penalty=0.6,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        early_stopping=True,
    )
    decoded_predictions = tokenizer.batch_decode(
        predictions.predictions, skip_special_tokens=True
    )
    dataset_pred = dataset.add_column("prediction", decoded_predictions)
    dataset_pred = dataset_pred.remove_columns(["attention_mask", "input_ids"])
    dataset_pred.save_to_disk(os.path.join(checkpoints, "predict_wiki-news"))
