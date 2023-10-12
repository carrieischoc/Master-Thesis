# Master-Thesis
Styled Text Summarization via Domain-Specific Paraphrasing
## Abstract
With the advent of deep learning techniques, numerous state-of-the-art summarization models have been proposed, each typically focusing on a specific domain. However, while these traditional methods effectively condense information, they often lack the flexibility to adjust the style of summaries. Summaries are often crafted in a conventional style or confined to the preferences of reviewers who provide the gold summaries for training models. There are cases where users prefer a differently-styled summary corresponding to relevant insights of a document.

Therefore, there is a need for effective summarization approaches that can generate styled summaries. This thesis focuses on the challenge of styled text summarization through domain-specific paraphrasing. Styled text summarization has the potential to cater to different expectations of users. It is particularly useful for communicating similar information to varied audiences, for instance, simplifying complex summaries from expert domains for the general public.

To this end, we propose a novel approach to styled text summarization that applies extractive training before domain-specific paraphrasing. It aims to not only condense the information but also adjust the summary style to suit specific domains, particularly for lengthy text. Our approach incorporates three distinct models: in-domain, domain- transfer, and cross-domain paraphrasing models. In-domain paraphrasing preserves the natural stayle of a specific domain, while re-writing the text into a more coherent final summary. Domain-transfer paraphrasing generates summaries in a style different from the the source domain, adapting to a particular target styling based in user preferences. The cross-domain model offers various summary styles for a broad range of domains.

The effectiveness of our approach is validated through experiments on English texts from three domains: Wikipedia, news, and legislative bills. The findings demonstrate that our models succeed in domain transfer of summaries, especially when the disparity in the quantity of training resources from different domains is not substantial. Supported by the evaluation phase, we outline further research prospects for this new view on generating stylistic summaries.

## Installation

To install the relevant external packages, please run `python3 -m pip install -r requirements.txt`.
In the case you are using some spacy-specific models, you may have to download additional dependencies by running `python3 -m spacy download en_core_web_sm` or `python3 -m spacy download en_core_web_md`.
