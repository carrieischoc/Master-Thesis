from LoadData import get_args
from inspection import get_print_lens, representative_len_samples
from similarity import get_print_simi

if __name__ == "__main__":
    args = get_args()

    if args.stats[0] == "length":

        get_print_lens(
            args.dataset[0],
            args.split[0],
            args.token_method[0],
            data_proportion=args.sample_propor,
        )

    if args.stats[0] == "similarity":
        get_print_simi(
            args.dataset[0], args.split[0], data_proportion=args.sample_propor
        )

    if args.stats[0] == "samples_len":
        representative_len_samples(args.dataset[0], args.split[0])
