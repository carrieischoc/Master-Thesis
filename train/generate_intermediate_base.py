import os
from preprocess.utils import base_path, get_args
from preprocess import OracleAlign, GreedyAlign


def generate_intermediate_baselines(
    dataset_name: str,
    split: str,
    method: str,
    base_path: str,
    num_process: int,
    optimization_attribute: str,
    topn: int = 1,
    sample_propor: float = 1.0,
):

    if method == "oracle":  # use the original dataset
        dataset_intermediate = OracleAlign.extract_similar_summaries(
            dataset_name=dataset_name,
            split=split,
            base_path=base_path,
            top_n=topn,
            optimization_attribute=optimization_attribute,
            num_proc=num_process,
            sample_propor=sample_propor,
        )
        path = os.path.join(
            base_path, args.dataset[0], split, f"baselines/intermediate_top{str(topn)}"
        )
        dataset_intermediate.save_to_disk(path)

    elif method == "greedy":  # make sure the format of reference/summary: list/str

        dataset_intermediate = GreedyAlign.extract_greedy_summaries(
            dataset_name=dataset_name,
            split=split,
            base_path=base_path,
            optimization_attribute=optimization_attribute,
            num_proc=num_process,
            sample_propor=sample_propor,
        )
        path = os.path.join(
            base_path, dataset_name, split, "baselines/intermediate_greedy"
        )
        dataset_intermediate.save_to_disk(path)


if __name__ == "__main__":

    base_path = base_path

    args = get_args()
    num_process = 16
    top_n = args.n_top

    if args.method[0] == "oracle":

        generate_intermediate_baselines(
            args.dataset[0],
            args.split[0],
            "oracle",
            base_path,
            num_process,
            "recall",
            top_n,
            args.sample_propor,
        )

    elif args.method[0] == "greedy":

        generate_intermediate_baselines(
            args.dataset[0],
            args.split[0],
            "greedy",
            base_path,
            num_process,
            "fmeasure",
            sample_propor=args.sample_propor,
        )
