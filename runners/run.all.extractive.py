import pandas

from bert_score import score

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

dataset = pandas.read_csv("../datasets/reviews.csv").truncate(after=config.SUMMARIES_CHUNK - 1)

original_texts = dataset['original_texts'].fillna(" ").to_list()
human_summaries = dataset['human_summaries'].fillna(" ").to_list()

# executing BERT
from extractive.bert.bert import BERT
dataset['bert'] = BERT().multi_apply(original_texts, verbose=True)

# executing KL_SUM
from extractive.kl_sum.kl_sum import KLSum
dataset['kl_sum'] = KLSum().multi_apply(original_texts, verbose=True)

# executing lexrank
from extractive.lexrank.lexrank import LexRank
dataset['lexrank'] = LexRank().multi_apply(original_texts, verbose=True)

# executing LSA
from extractive.lsa.lsa import LSA
dataset['lsa'] = LSA().multi_apply(original_texts, verbose=True)

# executing LUHN
from extractive.luhn.luhn import Luhn
dataset['luhn'] = Luhn().multi_apply(original_texts, verbose=True)

# executing LUHN
from extractive.textrank.textrank import TextRank
dataset['textrank'] = TextRank().multi_apply(original_texts, verbose=True)

dataset.to_csv('../outputs/extractive.csv', index=False)