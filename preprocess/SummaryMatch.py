from collections import namedtuple
from typing import NamedTuple
import operator
import heapq
import numpy as np
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
from .split import check_combine_str
from summaries.utils import get_nlp_model


def extract_similar_summaries(
    dataset, top_n: int = 3, match_n: int = 2, optimization_attribute: str = "fmeasure"
):
    """
    Extract top n sentences similar to summary sentences from reference.
    """

    dataset = check_combine_str(dataset, ["source", "target"])

    map_dict = {
        "top_n": top_n,
        "match_n": match_n,
        "optimization_attribute": optimization_attribute,
    }
    dataset = dataset.map(map_top_rouges_n_match, fn_kwargs=map_dict)

    return dataset


def top_rouges_n_match(
    summary: str,
    reference: str,
    top_n: int = 3,
    match_n: int = 2,
    optimization_attribute: str = "fmeasure",
) -> NamedTuple: # intermediate_summary: List[str]
    """
    Based on summaries module
    """

    # get doc format of summary and reference

    similar_sentences = namedtuple(
        "similar_sentences", "indices scores positions sentences reference summary"
    )
    similar_sentences.indices = []
    similar_sentences.scores = []
    similar_sentences.summary = []

    # get nlp model - shouldn't disable lemma, tokenizer...
    nlp = get_nlp_model(size="sm", disable=("ner",), lang="en")

    # combine n-grams and text sentence into one list comprehension loop
    reference_zip = [
        [_create_ngrams([token.lemma_ for token in sentence], n=match_n), sentence.text]
        for sentence in nlp(reference).sents
    ]
    reference_zip = list(zip(*reference_zip))
    reference_ngrams = reference_zip[0]
    reference_list = reference_zip[1]

    for sentence in nlp(summary).sents:

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
        topn = heapq.nlargest(top_n, enumerate(score), key=operator.itemgetter(1))
        similar_sentences.indices += list(zip(*topn))[0]
        similar_sentences.scores += list(zip(*topn))[1]
        similar_sentences.summary.append(sentence)

    # sort and remove duplicate indices to make summaries consistent
    sorted_scores = sorted(
        zip(set(similar_sentences.indices), similar_sentences.scores)
    )
    similar_sentences.indices = np.array(sorted_scores)[:, 0].astype(int)
    similar_sentences.scores = np.array(sorted_scores)[:, 1]

    similar_sentences.sentences = list(
        np.array(reference_list)[similar_sentences.indices]
    )
    similar_sentences.positions = np.array(similar_sentences.indices) / max(
        len(reference_list) - 1, 1
    )
    similar_sentences.reference = reference_list

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
    example["source"] = similar_sentences.reference
    example["target"] = similar_sentences.summary

    return example
