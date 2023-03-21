import argparse

base_path = "/home/jli/working_dir/2022-jiahui-li-thesis/data/"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():

    # get command arguments
    parser = argparse.ArgumentParser(description="Get training parameters of datasets")
    parser.add_argument("-d", "--dataset", nargs=1, type=str, help="name of dataset")
    parser.add_argument(
        "-m",
        "--method",
        nargs=1,
        type=str,
        choices=["oracle", "greedy", "oracle_score", "oracle_random", "lexrank", "L", "target_len"],
        help="method of intermediate summary",
    )
    parser.add_argument(
        "-n",
        "--n_top",
        nargs="?",
        type=int,
        const=1,
        default=1,
        help="oracle alignment coefficient",
    )
    parser.add_argument("--split", nargs=1, type=str, help="split of dataset")
    parser.add_argument(
        "-p",
        "--sample_propor",
        nargs="?",
        type=float,
        const=1.0,
        default=1.0,
        help="proportion of samples",
    )
    parser.add_argument("--debug", default=True, type=str2bool)
    parser.add_argument("--drop_ratio", default=False, type=str2bool)
    parser.add_argument(
        "--option",
        nargs=1,
        type=str,
        choices=[
        "extraction/lexrank_target_len",
        "extraction/lexrank_L",
        "baselines/intermediate_greedy",
        "baselines/intermediate_top1",
        "baselines/intermediate_top_extended_oracle_random",
        "baselines/intermediate_top_extended_oracle_score",
    ],
        help="training dataset options",
    )

    args = parser.parse_args()

    return args
