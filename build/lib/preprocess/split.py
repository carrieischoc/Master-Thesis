import spacy
from datasets import load_dataset
from spacy.language import Language


def combine_into_string(dataset):
    "Combine list into a string."
    dataset = dataset.map(lambda example: {'source': ' '.join(example['source']), 'target': ' '.join(example['target'])})
    return dataset

def split_into_sentences(dataset):
    '''
    Expected a string or 'Doc' as input.
    '''
    nlp = spacy.load('en_core_web_sm', disable=("tagger", "lemmatizer", "ner"))
    dataset = dataset.map(lambda example: {'source': [str(sent) for sent in nlp(example['source']).sents], 'target': [str(sent) for sent in nlp(example['target']).sents]})
    return dataset