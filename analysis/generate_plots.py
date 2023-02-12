import os
import seaborn as sns
from datasets import Dataset
from preprocess.utils import base_path

if __name__ == "__main__":

    split = 'train'
    dataset_name = "GEM/xwikis_en"
    path = os.path.join(base_path, dataset_name, split, "list_list_format")
    dataset = Dataset.load_from_disk(path)
    sns.stripplot(data=dataset["target_len"])