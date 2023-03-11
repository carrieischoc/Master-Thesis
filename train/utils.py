import os
from datasets import Dataset

def load_from_path(dataset_name: str, split: str, base_path: str, name):
    path = os.path.join(
        base_path, dataset_name, split, name
    )
    dataset = Dataset.load_from_disk(path)

    return dataset