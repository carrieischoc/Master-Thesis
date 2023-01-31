import argparse
from LoadData import load_data
from preprocess import OracleAlign, GreedyAlign


def get_args():

    # get command arguments
    parser = argparse.ArgumentParser(description="Get training parameters of datasets")
    parser.add_argument("-d", "--dataset", nargs=1, type=str, help="name of dataset")
    parser.add_argument("-m", "--method", nargs=1, type=str, help="method of intermediate summary")
    parser.add_argument("-n", "--n_top", nargs="?", type=int,const=1,
        default=1, help="oracle alignment coefficient")
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


def generate_intermediate_baselines(
    dataset,
    dataset_name: str,
    split: str,
    method: str,
    base_path: str,
    num_process: int,
    optimization_attribute: str,
    topn: int = 1,
):

    if method == "oracle":  # use the original dataset
        dataset_intermediate = OracleAlign.extract_similar_summaries(
            dataset,
            dataset_name=dataset_name,
            split=split,
            base_path=base_path,
            top_n=topn,
            optimization_attribute=optimization_attribute,
            num_proc=num_process,
        )
        dataset_intermediate.save_to_disk(
            base_path
            + args.dataset[0]
            + "/"
            + split
            + "/baselines/"
            + "intermediate_top"
            + str(topn)
        )

    elif method == "greedy":  # make sure the format of reference/summary: list/str

        dataset_intermediate = GreedyAlign.extract_greedy_summaries(
            dataset,
            dataset_name=dataset_name,
            split=split,
            base_path=base_path,
            optimization_attribute=optimization_attribute,
            num_proc=num_process,
        )
        dataset_intermediate.save_to_disk(
            base_path
            + args.dataset[0]
            + "/"
            + split
            + "/baselines/"
            + "intermediate_greedy"
        )


if __name__ == "__main__":

    base_path = "/home/jli/working_dir/2022-jiahui-li-thesis/data/"

    args = get_args()
    dataset = load_data(args.dataset[0], args.split[0], args.sample_propor)
    num_process = 16
    top_n = args.n_top

    if args.method[0] == 'oracle':
        
        generate_intermediate_baselines(
            dataset,
            args.dataset[0],
            args.split[0],
            "oracle",
            base_path,
            num_process,
            "recall",
            top_n,
        )
    
    elif args.method[0] == 'greedy':
        
        generate_intermediate_baselines(
            dataset,
            args.dataset[0],
            args.split[0],
            "greedy",
            base_path,
            num_process,
            "fmeasure"
        )
