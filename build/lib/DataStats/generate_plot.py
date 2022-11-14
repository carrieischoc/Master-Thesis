import os
from typing import List
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import median
from matplotlib import pyplot as plt
from LoadData import get_args, read_js
from inspection import get_lens
from similarity import get_simi

# create folder plots to store figures if not exist
os.makedirs("./plots/", exist_ok=True)

sns.set_theme(style="white", palette="pastel")
sns.despine(left=True)
paper_rc = {"lines.linewidth": 0.5, "lines.markersize": 15}
sns.set_context("paper", rc=paper_rc)


def convert_one_class_to_dataframe(
    ds: str, fe: str, cls: str, sts: List
) -> pd.DataFrame:
    df = pd.DataFrame()
    df[fe] = sts
    df = df.explode(fe, ignore_index=True)
    df["dataset"] = ds
    df["class"] = cls

    return df


def convert_classes_to_dataframe(
    stats_of_class: List, cls_names: List[str], ds_names: List[str], fe: str
) -> pd.DataFrame:
    df_list = []
    cls_n = len(cls_names)
    ds_n = len(ds_names)

    for i in range(ds_n):
        for j in range(cls_n):
            df_tmp = convert_one_class_to_dataframe(
                ds_names[i], fe, cls_names[j], stats_of_class[i * cls_n + j]
            )
            df_list.append(df_tmp)

    df = pd.concat(df_list, axis=0)
    df = df.reset_index(drop=True)

    return df


# Draw a nested violinplot and split the violins for easier comparison
def draw_violin(
    df: pd.DataFrame,
    name: str,
    y: str = "length",
    hue: str = "class",
    savefig: bool = True,
    showm: bool = True,
):
    _, ax = plt.subplots()

    sns.violinplot(
        data=df,
        x="dataset",
        y=y,
        hue=hue,
        showmeans=True,
        ax=ax,
        scale="count",
        cut=0,
        inner="quartile",
        dodge=0.6,
        palette="YlGnBu_d",
    )

    if showm:
        if y == "length":
            sns.pointplot(  # mean and std shown with bars
                data=df,
                x="dataset",
                y=y,
                hue=hue,
                dodge=True,
                capsize=0.15,
                ci="sd",
                join=False,
                ax=ax,
                scale=1.2,
                palette={"ref": "#76EE00", "sum": "#76EE00"},
            )
            sns.pointplot(  # estimator is median, shown without bars
                data=df,
                x="dataset",
                y=y,
                hue=hue,
                dodge=True,
                capsize=0.2,
                ci=0,
                join=False,
                estimator=median,
                ax=ax,
                scale=0,
                palette={"ref": "#FF6103", "sum": "#FF6103"},
            )
        elif y == "similarity":
            sns.pointplot(  # mean and std shown with bars
                data=df,
                x="dataset",
                y=y,
                hue=hue,
                dodge=0.6,
                capsize=0.15,
                ci="sd",
                join=False,
                ax=ax,
                scale=1.2,
                palette={"mean": "#76EE00", "max": "#76EE00", "min": "#76EE00"},
            )
            sns.pointplot(  # estimator is median, shown without bars
                data=df,
                x="dataset",
                y=y,
                hue=hue,
                dodge=0.6,
                capsize=0.2,
                ci=0,
                join=False,
                estimator=median,
                ax=ax,
                scale=0,
                palette={"mean": "#FF6103", "max": "#FF6103", "min": "#FF6103"},
            )
        else:
            raise ValueError("Invalid parameter for 'y' specified!")

    handles, labels = ax.get_legend_handles_labels()
    if y == "length":
        labels_n = 1
    if y == "similarity":
        labels_n = 3

    ax.legend(
        handles[0:labels_n],
        labels[0:labels_n],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )

    if savefig:
        plt.savefig(f"plots/{name}_violin.png", bbox_inches="tight", dpi=900)


def draw_strip(
    df: pd.DataFrame,
    name: str,
    logscale=True,
    y: str = "length",
    hue: str = "class",
    savefig: bool = True,
    showm: bool = True,
):
    _, ax = plt.subplots()

    sns.stripplot(
        data=df,
        x="dataset",
        y=y,
        hue=hue,
        dodge=True,
        size=2,
        jitter=True,
        ax=ax,
        palette="YlGnBu_d",
    )

    if showm:
        sns.pointplot(  # mean and std shown with bars
            data=df,
            x="dataset",
            y=y,
            hue=hue,
            capsize=0.15,
            ci="sd",
            join=False,
            ax=ax,
            scale=1.2,
            palette={"ref": "k", "sum": "k"},
        )
        sns.pointplot(  # estimator is median, shown without bars
            data=df,
            x="dataset",
            y=y,
            hue=hue,
            capsize=0.2,
            ci=0,
            join=False,
            estimator=median,
            ax=ax,
            scale=0,
            palette={"ref": "#FF6103", "sum": "#FF6103"},
        )

    if logscale:
        ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )

    if savefig:
        if logscale:
            plt.savefig(f"plots/{name}_log_strip.png", bbox_inches="tight", dpi=900)
        else:
            plt.savefig(f"plots/{name}_strip.png", bbox_inches="tight", dpi=900)


def draw_bar(
    df: pd.DataFrame,
    name: str,
    y: str = "length",
    hue: str = "class",
    savefig: bool = True,
    logscale: bool = True,
):
    _, ax = plt.subplots()

    sns.barplot(
        data=df,
        x="dataset",
        y=y,
        hue=hue,
        linewidth=0.5,
        ci="sd",
        capsize=0.1,
        palette="YlGnBu_d",
        ax=ax,
    )

    if logscale:
        ax.set_yscale("log")

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    if savefig:
        if logscale:
            plt.savefig(f"plots/{name}_log_bar.png", bbox_inches="tight", dpi=900)
        else:
            plt.savefig(f"plots/{name}_bar.png", bbox_inches="tight", dpi=900)


def plot_pos(pos: List, name: str, savefig: bool = True):
    _, ax = plt.subplots()

    sns.histplot(
        pos,
        stat="probability",
        kde=True,
        binrange=(0, 1),
        color="green",
        line_kws=dict(lw=1.5),
        bins=50,
        ax=ax,
    )
    plt.title(name)
    if savefig:
        plt.savefig(f"plots/{name}_pos.png")


def cal_show_CompressionRatio(
    sts: List, name: str, savefig: bool = True, logscale: bool = True
):
    _, ax = plt.subplots()

    compression_ratio = sts[0] / sts[1]
    bw = np.ceil(np.max(compression_ratio) / 25)

    sns.histplot(compression_ratio, color="green", binwidth=bw, ax=ax)
    ax.axvline(x=np.mean(compression_ratio), label="mean", color="red")
    ax.axvline(x=np.max(compression_ratio), label="max", color="green")
    ax.axvline(x=np.median(compression_ratio), label="median", color="k")
    ax.axvline(x=np.std(compression_ratio), label="std", color="gray", linestyle="--")
    plt.title(name)

    if logscale:
        ax.set_yscale("log")
    if savefig:
        if logscale:
            plt.savefig(f"plots/{name}_log_cpRatio.png", dpi=900)
        else:
            plt.savefig(f"plots/{name}_cpRatio.png", dpi=900)


if __name__ == "__main__":
    args = get_args()

    datasets = read_js("ds_name_list.json")
    stats_of_class = []

    if args.stats[0] == "length":
        cls_names = ["ref", "sum"]

        for dataset in datasets:
            sts = get_lens(
                dataset, args.split[0], args.token_method[0], args.sample_propor
            )
            stats_of_class.append(sts.src.lens)
            stats_of_class.append(sts.tg.lens)

            # Histogram of compression ratio (w/o log scale) with vertical lines (mean - red, max - green, median - black, std - dashed gray)
            cal_show_CompressionRatio(
                [sts.src.lens, sts.tg.lens], args.token_method[0] + "_len_" + dataset
            )
            cal_show_CompressionRatio(
                [sts.src.lens, sts.tg.lens],
                args.token_method[0] + "_len_" + dataset,
                logscale=False,
            )

        df = convert_classes_to_dataframe(stats_of_class, cls_names, datasets, "length")

        # draw all datasets in one of (w/o logscale) stripplot marked with mean and std (black), median (orange).
        draw_strip(df, args.token_method[0] + "_len", logscale=False)
        draw_strip(df, args.token_method[0] + "_len")

        # draw violin plot of reference/summary marked with mean and std (green), median (orange).
        draw_violin(df[df["class"] == "ref"], args.token_method[0] + "_len_ref")
        draw_violin(df[df["class"] == "sum"], args.token_method[0] + "_len_sum")

    if args.stats[0] == "similarity":
        cls_names = ["mean", "max", "min"]

        for dataset in datasets:
            sts = get_simi(dataset, args.split[0], args.sample_propor)
            stats_of_class.append(sts.mean)
            stats_of_class.append(sts.max)
            stats_of_class.append(sts.min)

            # Plot of distribution of relative position of most similar sentence in article
            plot_pos(sts.pos, dataset)

        df = convert_classes_to_dataframe(
            stats_of_class, cls_names, datasets, "similarity"
        )

        # draw violin plot of distribution of mean, max, min of similarity, with mean and std (green), median (red)
        draw_violin(df, "simi", y="similarity")
