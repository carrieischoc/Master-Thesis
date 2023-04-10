import os
import math
import seaborn as sns
from matplotlib import pyplot as plt
import inspection
from preprocess.utils import base_path
from train.utils import load_from_path

sns.set_theme(style="white", palette="pastel")
sns.despine(left=True)
paper_rc = {"lines.linewidth": 0.5, "lines.markersize": 15}
sns.set_context("paper", rc=paper_rc)


def plot_one_feature(data, dataset_name: str, split: str, x: str, y: str, feature: str):
    sns.violinplot(data=data, y=y, showmeans=True, inner="quartile", cut=0)
    plt.xlabel(x)
    path = os.path.join("plots", dataset_name, split, feature)
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{y}.png", bbox_inches="tight", dpi=1080)


def input_output_length(dataset_name: str, split: str, base_path: str, feature: str):
    """
    feature: different options for length stats.
    """
    dataset = load_from_path(dataset_name, split, base_path, feature)
    if feature == "list_str_format":  # target;
        feature_name = "target"
    else:  # intermediate summary
        feature_name = "intermediate_summary"

    df = dataset.to_pandas()
    if dataset.features[feature_name]._type != "Value":
        df[feature_name] = df[feature_name].str.join(" ")
    stats = inspection.whitespace_token(df[feature_name])
    df["tokens_num"] = stats.lens
    print("*" * 8, feature, "*" * 8)
    print(f"Stats for the number of the tokens of {feature_name} for {feature}:")
    print(
        f"Mean: {math.ceil(stats.mean)}, Median: {math.ceil(stats.median)}, STD: {math.ceil(stats.std)}."
    )
    plot_one_feature(
        df, dataset_name, split, x=feature_name, y="tokens_num", feature=feature
    )


if __name__ == "__main__":

    split = "train"
    dataset_name = "GEM/xwikis_en"

    # options: "list_str_format", "extraction/lexrank_target_len", "extraction/lexrank_L"
    # "baselines/intermediate_greedy", "baselines/intermediate_top1",
    # "baselines/intermediate_top_extended_oracle_random", 
    # "baselines/intermediate_top_extended_oracle_score"
    options = [
        "list_str_format",
        "extraction/lexrank_target_len",
        "extraction/lexrank_L",
        "baselines/intermediate_greedy",
        "baselines/intermediate_top1",
        "baselines/intermediate_top_extended_oracle_random",
        "baselines/intermediate_top_extended_oracle_score",
    ]
    for op in options:
        input_output_length(dataset_name, split, base_path, op)
