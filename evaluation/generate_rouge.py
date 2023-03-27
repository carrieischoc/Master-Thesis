import os
import evaluate
from datasets import Dataset
from preprocess.utils import base_path, get_args

evaluate.list_evaluation_modules()





if __name__ == "__main__":

    args = get_args()

    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    def compute_score(example):
        example["bleu"] = bleu.compute(
            predictions=example["prediction"], references=example["target"]
        )
        example["rouge"] = rouge.compute(
            predictions=example["prediction"], references=example["target"]
        )
        example["meteor"] = meteor.compute(
            predictions=example["prediction"], references=example["target"]
        )

        return example
    
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
        path = os.path.join(cp, "predict")
        dataset = Dataset.load_from_disk(path) 
        dataset = dataset.map(compute_score, num_proc=8)
        dataset.save_to_disk(path)