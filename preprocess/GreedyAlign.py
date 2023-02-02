import os
from typing import List, Counter, Union, NamedTuple
from collections import namedtuple
import numpy as np
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
from datasets import Dataset
from spacy.tokens import Span
from spacy.language import Doc
from summaries.utils import get_nlp_model
from LoadData import load_data
from .split import check_split_sent, check_combine_feature


def compute_rouge_score(
    current_alignment: Union[Span, Doc],
    summary_ngrams: Counter,
    optimization_attribute: str,
    n: int,
) -> NamedTuple:  # intermediate_summary: List[str]

    current_align_ngrams = _create_ngrams(
        [token.lemma_ for token in current_alignment], n=n
    )
    score = getattr(
        _score_ngrams(
            target_ngrams=summary_ngrams, prediction_ngrams=current_align_ngrams
        ),
        optimization_attribute,
    )
    return score


def greedy_alignment(
    reference: List[str],
    summary: str,
    n: int = 2,
    optimization_attribute="fmeasure",
    lang="en",
):
    prev_score = 0
    current_alignment = ""
    selected_flag = np.zeros(len(reference))  # 0: not selected
    greedy_alignment = namedtuple("greedy_alignment", "scores sentences n_gram")
    # get nlp model - shouldn't disable lemma, tokenizer...
    nlp = get_nlp_model(size="sm", disable=("ner",), lang=lang)
    summary_ngrams = _create_ngrams([token.lemma_ for token in nlp(summary)], n=n)

    # find the optimal approximation to the entire set of
    # all sentences in the gold summary.

    while np.any(selected_flag == 0):
        new_best_score = 0
        new_best_index = -1
        new_best_hypothesis_sentence = ""
        # Try the combination of the previously selected sentences
        # + any of the remaining ones
        for i in range(len(selected_flag)):
            if not selected_flag[i]:
                rouge_score = compute_rouge_score(
                    nlp(current_alignment + " " + reference[i]),
                    summary_ngrams,
                    optimization_attribute,
                    n,
                )

                if rouge_score > new_best_score:
                    new_best_score = rouge_score
                    new_best_hypothesis_sentence = reference[i]
                    new_best_index = i

        # no optimal solutions, all scores = 0, degrade to 1-gram match
        if new_best_score == 0 and prev_score == 0:
            if n == 1:  # no matched 1-gram
                return greedy_alignment(prev_score, [], n)
            else:
                n = 1
                summary_ngrams = _create_ngrams(
                    [token.lemma_ for token in nlp(summary)], n=1
                )
        # Additional sentence was no longer improving the score; terminal condition
        elif new_best_score <= prev_score:
            sorted_selected_indices = sorted(np.where(selected_flag == 1)[0])
            current_alignment_list = list(np.array(reference)[sorted_selected_indices])
            # record the n of ROUGE-n
            return greedy_alignment(prev_score, current_alignment_list, n)
        else:
            # Update hypothesis
            current_alignment += " " + new_best_hypothesis_sentence
            prev_score = new_best_score

            # Also remove this sentence from the candidate set
            # so it cannot be added in future iterations
            selected_flag[new_best_index] = 1

    # if all source sentences are selected, keep the original source/target??
    return greedy_alignment(prev_score, reference, n)


def map_greedy_alignment(example, match_n, optimization_attribute, lang):

    greedy_summary = greedy_alignment(
        example["source"], example["target"], match_n, optimization_attribute, lang
    )

    # generate intermediate summary from greedy method that maximize ROUGE scores
    example["intermediate_summary"] = greedy_summary.sentences
    example["intermediate_summary_scores"] = greedy_summary.scores
    example["n_gram"] = greedy_summary.n_gram

    return example


def extract_greedy_summaries(
    dataset_name: str,
    split: str,
    base_path: str,
    match_n: int = 2,
    optimization_attribute: str = "fmeasure",
    lang: str = "en",
    num_proc: int = 16,
    sample_propor: float = 1.0,
):
    """
    The set of selected sentences is maximized with respect to the entire gold summary.
    """

    try:
        path_str = os.path.join(base_path, dataset_name, split, "list_str_format")
        dataset = Dataset.load_from_disk(path_str)

    except FileNotFoundError:
        # try to make use of results in oracle alignment
        try:
            path_list = os.path.join(base_path, dataset_name, split, "list_list_format")
            dataset_ls = Dataset.load_from_disk(path_list)
            dataset = check_combine_feature(dataset_ls, "target")
        except FileNotFoundError:
            dataset = load_data(dataset_name, split, sample_propor)
            # summary must be a string
            dataset = check_combine_feature(dataset, "target")
            # reference must be a list
            dataset = check_split_sent(dataset, ["source"])

        dataset.save_to_disk(path_str)

    map_dict = {
        "match_n": match_n,
        "optimization_attribute": optimization_attribute,
        "lang": lang,
    }
    dataset = dataset.map(map_greedy_alignment, fn_kwargs=map_dict, num_proc=num_proc)

    return dataset
