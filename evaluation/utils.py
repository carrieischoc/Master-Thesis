import os
from datasets import Dataset


def get_domain_specific_tokenized_testdata(
    tokenizer,
    base_path: str,
    dataset_name: str,
    options: str,
    domain: str,
    max_input_length: int = 832,
    max_output_length: int = 128,
):
    """
    Add specific domain mark at the beginning of each sample;
    It could be different domains for each testing data set.
    """

    path_t = os.path.join(
        base_path,
        dataset_name,
        "test",
        "tokenized",
        options,
        f"{str(max_input_length)}_{str(max_output_length)}_{domain}",
    )
    try:
        dataset = Dataset.load_from_disk(path_t)

    except FileNotFoundError:
        path_ds = os.path.join(
            base_path,
            dataset_name,
            "test",
            "tokenized",
            options,
            domain,
        )
        try:
            dataset = Dataset.load_from_disk(path_ds)
        except FileNotFoundError:
            domain_dic = {
                "wikis": "Wikipedia Text: ",
                "bills": "Bills Text: ",
                "news": "News Text: ",
            }
            dataset = Dataset.load_from_disk(
                os.path.join(
                    base_path,
                    dataset_name,
                    "test",
                    "tokenized",
                    options,
                    "noise_removed",
                )
            )
            dataset = dataset.map(
                lambda example: {
                    "intermediate_summary": domain_dic[domain]
                    + example["intermediate_summary"]
                },
                num_proc=16,
            )
            dataset.save_to_disk(path_ds)

        def tokenize_function(example):
            # Tokenize inputs and targets
            inputs = tokenizer(
                example["intermediate_summary"],
                padding="max_length",
                truncation=True,
                max_length=max_input_length,
            )

            # Return the input and target IDs and the attention masks
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

        dataset = dataset.map(
            tokenize_function,
            num_proc=16,
            remove_columns=["intermediate_summary", "target"],
        )
        dataset.save_to_disk(path_t)

    return dataset
