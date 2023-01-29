from typing import List
from summaries.utils import get_nlp_model


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


def split_into_sentences(dataset, feature: str):
    nlp = get_nlp_model(size="sm", disable=("ner",), lang="en")
    dataset = dataset.map(
        lambda example: {feature: [sent.text for sent in nlp(example[feature]).sents]}
    )
    return dataset


def check_split_feature(dataset, feature: str):

    # get nlp model - shouldn't disable lemma, tokenizer...
    # distinguish list of paragraphs from list of sentences
    if dataset.features[feature]._type == "Sequence":
        for x in dataset.column_names:
            if 'section_level' in x:
                dataset = combine_into_string(dataset, feature)
                dataset = dataset.remove_columns(x)

    # transform string to list of sentences
    if dataset.features[feature]._type == "Value":
        dataset = split_into_sentences(dataset, feature)

    return dataset


def check_split_sent(dataset, feature: List[str]):

    for fe in feature:
        dataset = check_split_feature(dataset, fe)

    return dataset
