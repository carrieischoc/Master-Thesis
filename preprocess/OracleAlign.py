import os
from collections import namedtuple
from typing import NamedTuple, List
import numpy as np
from datasets import Dataset
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
from summaries.utils import get_nlp_model
from LoadData import load_data
from .split import check_split_sent


def extract_similar_summaries(
    dataset_name: str,
    split: str,
    base_path: str,
    top_n: int = 3,
    match_n: int = 2,
    optimization_attribute: str = "fmeasure",
    num_proc: int = 16,
    sample_propor: float = 1.0,
):
    """
    Extract top n sentences similar to summary sentences from reference.
    """

    try:
        path = os.path.join(base_path, dataset_name, split, "list_list_format")
        dataset = Dataset.load_from_disk(path)

    except FileNotFoundError:
        # check and generate ref[List], sum[List].
        dataset = load_data(dataset_name, split, sample_propor)
        dataset = check_split_sent(dataset, ["source", "target"])
        dataset.save_to_disk(path)

    map_dict = {
        "top_n": top_n,
        "match_n": match_n,
        "optimization_attribute": optimization_attribute,
    }
    dataset = dataset.map(map_top_rouges_n_match, fn_kwargs=map_dict, num_proc=num_proc)

    return dataset


def top_rouges_n_match(
    summary: List[str],
    reference: List[str],
    top_n: int = 3,
    match_n: int = 2,
    optimization_attribute: str = "fmeasure",
) -> NamedTuple:  # intermediate_summary: List[str]
    """
    Based on summaries module.
    Compasions are carried out between ref[List] and sum[List].
    """

    # it's faster than passing nlp model as arguments.
    # get nlp model - shouldn't disable lemma, tokenizer...
    nlp = get_nlp_model(size="sm", disable=("ner",), lang="en")
    # get doc format (spacy) of summary and reference
    reference_doc = [nlp(sentence) for sentence in reference]
    summary_doc = [nlp(sentence) for sentence in summary]

    similar_sentences = namedtuple(
        "similar_sentences", "indices scores positions sentences"
    )
    similar_sentences.indices = []
    similar_sentences.scores = []

    # generate reference n-grams
    reference_ngrams = [
        _create_ngrams([token.lemma_ for token in sentence], n=match_n)
        for sentence in reference_doc
    ]

    for sentence in summary_doc:

        target_ngrams = _create_ngrams([token.lemma_ for token in sentence], n=match_n)

        score = [
            getattr(
                _score_ngrams(
                    target_ngrams=target_ngrams, prediction_ngrams=source_ngrams
                ),
                optimization_attribute,
            )
            for _, source_ngrams in enumerate(reference_ngrams)
        ]

        # use heapq to sort (for long lists)
        # topn = heapq.nlargest(top_n, enumerate(score), key=operator.itemgetter(1))
        # similar_sentences.indices += list(zip(*topn))[0]
        # similar_sentences.scores += list(zip(*topn))[1]
        # remove duplicate indices and keep a replacement
        # set() is unordered and can change the order.
        # replace set with np.in1d
        sorted_indices = np.argsort(score)
        indices = list(
            sorted_indices[~np.in1d(sorted_indices, similar_sentences.indices)]
        )[-top_n:]
        similar_sentences.indices += indices
        similar_sentences.scores += list(np.array(score)[indices])

    # sort to make summaries consistent
    sorted_scores = sorted(zip(similar_sentences.indices, similar_sentences.scores))
    similar_sentences.indices = np.array(sorted_scores)[:, 0].astype(int)
    similar_sentences.scores = np.array(sorted_scores)[:, 1]

    similar_sentences.sentences = list(np.array(reference)[similar_sentences.indices])
    similar_sentences.positions = np.array(similar_sentences.indices) / max(
        len(reference) - 1, 1
    )

    return similar_sentences


def map_top_rouges_n_match(example, top_n, match_n, optimization_attribute):

    similar_sentences = top_rouges_n_match(
        example["target"],
        example["source"],
        top_n,
        match_n,
        optimization_attribute,
    )

    # generate intermediate summary from most similar sentences relative to target
    # transform all examples into list of sentences
    example["intermediate_summary"] = similar_sentences.sentences
    example["intermediate_summary_scores"] = similar_sentences.scores
    example["intermediate_summary_pos"] = similar_sentences.positions
    example["intermediate_summary_indices"] = similar_sentences.indices

    return example
