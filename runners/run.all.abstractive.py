import pandas

from bert_score import score

from transformers import logging
logging.set_verbosity_error()

dataset = pandas.read_csv("../datasets/reviews.csv").truncate(after=4999)

original_texts = dataset['original_texts'].fillna(" ").to_list()
human_summaries = dataset['human_summaries'].fillna(" ").to_list()

import sys
sys.path.append('..')

# executing BERT
from abstractive.BART.BART import BART
dataset['bart'] = BART().multi_apply(original_texts, verbose=True)

# executing KL_SUM
from abstractive.T5.T5 import T5
dataset['kl_sum'] = T5().multi_apply(original_texts, verbose=True)

dataset.to_csv('../outputs/abstractive.csv', index=False)