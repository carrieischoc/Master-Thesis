# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""


import csv
import json
import os

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{perez2021models,
  title={Models and Datasets for Cross-Lingual Summarisation},
  author={Perez-Beltrachini, Laura and Lapata, Mirella},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={9408--9423},
  year={2021}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
The XWikis Corpus (Perez-Beltrachini and Lapata, 2021) provides datasets with different language pairs and directions for cross-lingual abstractive document summarisation. This current version includes four languages: English, German, French, and Czech. The dataset is derived from Wikipedia. It is based on the observation that for a Wikipedia title, the lead section provides an overview conveying salient information, while the body provides detailed information. It thus assumes the body and lead paragraph as a document-summary pair. Furthermore, as a Wikipedia title can be associated with Wikipedia articles in various languages, 1) Wikipedia’s Interlanguage Links are used to find titles across languages and 2) given any two related Wikipedia titles, e.g., Huile d’Olive (French) and Olive Oil (English), the lead paragraph from one title is paired with the body of the other to derive cross-lingual pairs.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://datashare.ed.ac.uk/handle/10283/4188"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "CC BY-SA 4.0"

VERSION = datasets.Version("0.1.0")

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

LPAIRS = [
    f"{x}-{y}" for x in ["en","fr","cs","de"] for y in ["en","fr","cs","de"] if x!=y
] + ["en","fr","cs","de"]

_URLs = {
    "train": [ f"./train/{xy}.jsonl" for xy in LPAIRS]
    #     "./train/en-fr.jsonl",
    #     "./train/fr-en.jsonl",
    #     "./train/en-de.jsonl",
    #     "./train/de-en.jsonl",
    #     "./train/en-cs.jsonl",
    #     "./train/cs-en.jsonl",
    #     "./train/fr-de.jsonl",
    #     "./train/de-fr.jsonl",
    #     "./train/fr-cs.jsonl",
    #     "./train/cs-fr.jsonl",
    #     "./train/de-cs.jsonl",
    #     "./train/cs-de.jsonl",
    # "validation": [ f"./valid/{xy}.jsonl" for xy in LPAIRS],
    # "test": [ f"./test/{xy}.jsonl" for xy in LPAIRS if "-en" in xy],
    }


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class XWikis(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""


    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [datasets.BuilderConfig(name=xy , version=VERSION, description=f"XWikis. Language pair: {xy}") for xy in LPAIRS]

    DEFAULT_CONFIG_NAME = "fr-en"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = datasets.Features(
            {
                "gem_id": datasets.Value("string"),
                "gem_parent_id": datasets.Value("string"),
                "id": datasets.Value("string"),
                "src_title": datasets.Value("string"),
                "tgt_title": datasets.Value("string"),
                "src_document": datasets.features.Sequence(
                    {
                        "title": datasets.Value("string"),
                        "section_level": datasets.Value("string"),
                        "content": datasets.Value("string"),
                    }),
                "src_summary": datasets.Value("string"),
                "tgt_summary": datasets.Value("string")
                
                # These are the features of your dataset like images, labels ...
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_urls = {sp:f"./{sp[:5]}/{self.config.name}.jsonl" for sp in ["train"] }
        d_conf = dl_manager.download_and_extract(my_urls)
        ### TODO
        challenge_sets = []
        # challenge_sets = [
        #     ("challenge_test_abstractivity_%d" % (lvl), fname) \
        #         for lvl,fname in enumerate(d_conf["cs_abs"])
        # ] + [
        #     ("challenge_test_topic_diversity_%d" % (lvl), fname) \
        #         for lvl,fname in enumerate(d_conf["cs_abs"])
        # ]
        

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": d_conf["train"],
                    "split": "train",
                },
            #     datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": d_conf["validation"],
            #         "split": "validation"
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": d_conf["test"],
            #         "split": "test",
            #     },
            # ),
            )
        ] + [
                datasets.SplitGenerator(
                    name=challenge_split,
                    gen_kwargs={
                        "filepath": filename,
                        "split": challenge_split,
                    },
                )
                for challenge_split, filename in challenge_sets
            ]

    def _generate_examples(
        self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        with open(filepath, encoding="utf-8") as f:
            for row in f:
                data = json.loads(row)
                id_ = data["id"]
                # data["gem_parent_id"] = "GEM-wiki_cat_sum-%s-%d" % (split,data["id"]+1)
                # data["gem_id"] = "GEM-wiki_cat_sum-%s-%d" % (split,data["id"]+1)
                data["gem_parent_id"] = f"{self.config.name}-{split}-{id_}"
                data["gem_id"] = f"{self.config.name}-{split}-{id_}"
                yield id_,data
