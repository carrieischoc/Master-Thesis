import os
import json
import evaluate
from datasets import Dataset
from preprocess.utils import base_path, get_args

# from datasets import load_metric


if __name__ == "__main__":

    args = get_args()

    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # def compute_score(example):
    #     example["bleu"] = bleu.compute(
    #         predictions=example["prediction"], references=example["target"]
    #     )
    #     example["rouge"] = rouge.compute(
    #         predictions=example["prediction"], references=example["target"]
    #     )
    #     example["meteor"] = meteor.compute(
    #         predictions=example["prediction"], references=example["target"]
    #     )

    #     return example

    path_to_checkpoint = os.path.join(
        base_path,
        args.dataset[0],
        f"models/vanilla/{args.option[0]}_{str(args.drop_ratio)}_128",
    )
    # checkpoints = [
    #     os.path.join(path_to_checkpoint, f)
    #     for f in os.listdir(path_to_checkpoint)
    #     if "checkpoint" in f
    # ]
    # checkpoints = [
    #     os.path.join(path_to_checkpoint, "checkpoint-" + i)
    #     for i in ["1", "2", "3", "4", "5"]
    # ]
    # the best checkpoint
    checkpoints = [os.path.join(path_to_checkpoint, "checkpoint-5")]

    for cp in checkpoints:
        path = os.path.join(cp, "predict_2")
        dataset = Dataset.load_from_disk(path)
        # dataset = dataset.map(compute_score, num_proc=8)
        # blue_scores = bleu.compute(
        #     predictions=dataset["prediction"], references=dataset["target"]
        # )
        # f1 score
        rouge_scores = rouge.compute(
            predictions=dataset["prediction"],
            references=dataset["target"],
            use_aggregator=False,
        )
        # meteor_scores = meteor.compute(
        #     predictions=dataset["prediction"], references=dataset["target"]
        # )
        f = open(f"{path}/rouge", "w+")
        json.dump(rouge_scores, f)
