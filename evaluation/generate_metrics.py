import os
import evaluate
from datasets import Dataset
from preprocess.utils import base_path, get_args
from datasets import load_metric

# rouge = evaluate.load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# results = rouge.compute(predictions=predictions,references=references)
# print(results)





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
        f"models/vanilla/{args.option[0]}_{str(args.drop_ratio)}",
    )
    checkpoints = [
        os.path.join(path_to_checkpoint, f)
        for f in os.listdir(path_to_checkpoint)
        if "checkpoint" in f
    ]

    for cp in checkpoints:
        path = os.path.join(cp, "results_gpu")
        dataset = Dataset.load_from_disk(path) 
        # dataset = dataset.map(compute_score, num_proc=8)
        blue_scores = bleu.compute(
            predictions=dataset["prediction"], references=dataset["target"]
        )
        rouge_scores = rouge.compute(
            predictions=dataset["prediction"], references=dataset["target"]
        )
        meteor_scores = meteor.compute(
            predictions=dataset["prediction"], references=dataset["target"]
        )
        dataset.save_to_disk(path)