from typing import List, Counter, Union
from collections import namedtuple
import numpy as np
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
from spacy.tokens import Span
from spacy.language import Doc
from summaries.aligners import RougeNAligner
from .split import check_split_sent
from summaries.utils import get_nlp_model

def compute_rouge_score(
    current_alignment: Union[Span, Doc], summary_ngrams: Counter, optimization_attribute: str, n: int
):

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
    summary: List[str],
    n: int = 2,
    optimization_attribute="fmeasure",
    lang="en",
):
    prev_score = 0
    current_alignment = ""
    selected_flag = np.zeros(len(reference))  # 0: not selected
    greedy_alignment = namedtuple("greedy_alignment", "scores sentences")
    # get nlp model - shouldn't disable lemma, tokenizer...
    nlp = get_nlp_model(size="sm", disable=("ner",), lang="en")

    # get start by sentences with the maximum ROUGE scores relative to each sentence in the summary
    aligner = RougeNAligner(
        n=n, optimization_attribute=optimization_attribute, lang=lang
    )
    first_aligned_sentence = aligner.extract_source_sentences(summary, reference)[0]
    current_alignment += ' ' + first_aligned_sentence.sentence
    selected_index = first_aligned_sentence.index
    selected_flag[selected_index] = 1
    summary = " ".join(summary)
    summary_ngrams = _create_ngrams([token.lemma_ for token in nlp(summary)], n=n)
    prev_score = compute_rouge_score(
        nlp(current_alignment), summary_ngrams, optimization_attribute, n
    )

    while True:
        new_best_score = 0
        new_best_index = 0
        new_best_hypothesis_sentence = ""
        # Try the combination of the previously selected sentences + any of the remaining ones
        for i in range(len(selected_flag)):
            if not selected_flag[i]:
                rouge_score = compute_rouge_score(
                    nlp(current_alignment + ' ' + reference[i]),
                    summary_ngrams,
                    optimization_attribute,
                    n,
                )

                if rouge_score > new_best_score:
                    new_best_score = rouge_score
                    new_best_hypothesis_sentence = reference[i]
                    new_best_index = i

        # Additional sentence was no longer improving the score; terminal condition
        if new_best_score < prev_score:
            return greedy_alignment(prev_score, current_alignment)
        else:
            # Update hypothesis
            current_alignment += ' ' + new_best_hypothesis_sentence
            prev_score = new_best_score

            # Also remove this sentence from the candidate set so it cannot be added in future iterations
            selected_flag[new_best_index] = 1


def map_greedy_alignment(example, match_n, optimization_attribute, lang):

    greedy_summary = greedy_alignment(
        example["target"], example["source"], match_n, optimization_attribute, lang
    )

    example["greedy_summary"] = greedy_summary.sentences
    example["greedy_summary_scores"] = greedy_summary.scores

    return example


def extract_greedy_summaries(
    dataset, match_n: int = 2, optimization_attribute: str = "fmeasure", lang="en"
):
    """
    The set of selected sentences is maximized with respect to the entire gold summary.
    """

    dataset = check_split_sent(dataset, ["source", "target"])

    map_dict = {
        "match_n": match_n,
        "optimization_attribute": optimization_attribute,
        "lang": lang,
    }
    dataset = dataset.map(map_greedy_alignment, fn_kwargs=map_dict)

    return dataset
