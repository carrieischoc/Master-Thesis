from datasets import load_dataset
import inspection 
import LoadData
from preprocess import SummaryMatch

if __name__ == "__main__":

    xwiki_en = load_dataset('GEM/xwikis', 'en', split='train[:1%]')
    xwiki_en = xwiki_en.flatten()
    xwiki_en = LoadData.rename_datasets(xwiki_en)
    # inspection.get_print_lens("GEM/xwikis_en", data_proportion=0.1)
    xwiki_en_intermediate = SummaryMatch.extract_similar_summaries(xwiki_en)
    print(xwiki_en_intermediate[10])