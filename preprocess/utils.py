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
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    args = parser.parse_args()

    return args
