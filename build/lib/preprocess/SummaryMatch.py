from collections import namedtuple
from typing import List, NamedTuple
import operator
import heapq
import numpy as np
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
import split


def extract_similar_summaries(dataset, n: int=2, optimization_attribute: str = "recall"):
    '''
    Extract top n sentences similar to summary sentences from reference.
    '''

    if dataset.features["source"]._type == "Value":  # One row of source is one article.
        pass

    elif (
        dataset.features["source"]._type == "Sequence"
    ):  # Lists with split sentences or paragraphs
        if dataset.features["source"].feature._type == "Value":
            dataset = split.combine_into_string(dataset) # combine list elements into a string
        else:
            raise TypeError("Unknown type of source and target!")

    dataset = split.split_into_sentences(dataset)

    match_n_col = np.ones(dataset.num_rows,dtype=int)*n
    optimization_attribute_col = np.full(dataset.num_rows, optimization_attribute)
  
    dataset = dataset.add_column("match_n", match_n_col)
    dataset = dataset.add_column("optimization_attribute", optimization_attribute_col)
    dataset = dataset.map(map_top_rouges_n_match)

    return dataset


def top_rouges_n_match(summary: List[str], reference: List[str], n: int=2, optimization_attribute: str = "recall") -> NamedTuple:
    '''
    Based on summaries module
    '''

    similar_sentences = namedtuple("similar_sentences", "indices scores positions sentences")
    similar_sentences.indices = []
    similar_sentences.scores = []

    reference_ngrams = [_create_ngrams([token.lemma_ for token in sentence], n=n)
                            for sentence in reference]

    for sentence in summary:

        target_ngrams = _create_ngrams([token.lemma_ for token in sentence], n=n)

        score = [getattr(_score_ngrams(target_ngrams=target_ngrams, prediction_ngrams=source_ngrams), optimization_attribute) for _, source_ngrams in enumerate(reference_ngrams)]
        
        # use heapq to sort (for long lists)
        topn = heapq.nlargest(3, enumerate(score), key=operator.itemgetter(1))
        similar_sentences.indices += list(zip(*topn))[0]
        similar_sentences.scores += list(zip(*topn))[1]

    # sort and remove duplicate indices to make summaries consistent
    sorted_scores = sorted(zip(set(similar_sentences.indices),similar_sentences.scores))
    similar_sentences.indices = np.array(sorted_scores)[:,0].astype(int)
    similar_sentences.scores = np.array(sorted_scores)[:,1]

    similar_sentences.sentences = list(np.array(reference)[similar_sentences.indices])
    similar_sentences.positions = np.array(similar_sentences.indices) / max(len(reference) - 1, 1)

    return similar_sentences

def map_top_rouges_n_match(example):

    similar_sentences = top_rouges_n_match(example["target"], example["source"], example["match_n"], example["optimization_attribute"])

    example["similar_summary"] = similar_sentences.sentences
    example["similar_summary_scores"] = similar_sentences.scores
    example["similar_summary_pos"] = similar_sentences.positions

    return example
