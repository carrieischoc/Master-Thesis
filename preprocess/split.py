from typing import List
import spacy
from datasets import load_dataset
from spacy.language import Language


def combine_into_string(dataset, feature: str):
    "Combine list into a string."
    dataset = dataset.map(lambda example: {feature: " ".join(example[feature])})
    return dataset


def check_combine_feature(dataset, feature: str):
    if dataset.features[feature]._type == "Value":  # One row of feature is one article.
        pass

    elif (
        dataset.features[feature]._type == "Sequence"
    ):  # Lists with split sentences or paragraphs
        if dataset.features[feature].feature._type == "Value":
            dataset = combine_into_string(
                dataset, feature
            )  # combine list elements into a string

        else:
            raise TypeError("Unknown type of feature and target!")

    return dataset


def check_combine_str(dataset, feature: List[str]):

    for fe in feature:
        dataset = check_combine_feature(dataset, fe)

    return dataset


def split_into_sentences(dataset):
    """
    Expected a string or 'Doc' as input.
    """
    nlp = spacy.load("en_core_web_sm", disable=("tagger", "lemmatizer", "ner"))
    dataset = dataset.map(
        lambda example: {
            "source": [str(sent) for sent in nlp(example["source"]).sents],
            "target": [str(sent) for sent in nlp(example["target"]).sents],
        }
    )
    return dataset
