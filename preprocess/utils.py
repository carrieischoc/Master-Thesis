import argparse

base_path = "/home/jli/working_dir/2022-jiahui-li-thesis/data/"


def get_args():

    # get command arguments
    parser = argparse.ArgumentParser(description="Get training parameters of datasets")
    parser.add_argument("-d", "--dataset", nargs=1, type=str, help="name of dataset")
    parser.add_argument(
        "-m",
        "--method",
        nargs=1,
        type=str,
        choices=["oracle", "greedy", "oracle_score", "oracle_random", "lexrank"],
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

    args = parser.parse_args()

    return args
