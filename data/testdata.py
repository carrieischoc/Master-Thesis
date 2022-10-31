import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, Value, Sequence
from summaries.aligners import RougeNAligner
import inspection 

if __name__ == "__main__":

    xwiki_en = load_dataset('GEM/xwikis', 'en', split='train')
    inspection.get_print_lens("GEM/xwikis_en", data_proportion=0.1)