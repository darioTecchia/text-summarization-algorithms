import pandas

import os

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

def run_all(input_path, output_path):
    print('reading ' + os.path.abspath(input_path))
    dataset = pandas.read_csv(input_path).truncate(after=config.SUMMARIES_CHUNK - 1)

    original_texts = dataset['original_texts'].fillna(" ").to_list()

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

    # executing textrank
    # from extractive.textrank.textrank import TextRank
    # dataset['textrank'] = TextRank().multi_apply(original_texts, verbose=True)

    # executing BERT
    from abstractive.BART.BART import BART
    dataset['bart'] = BART().multi_apply(original_texts, verbose=True)

    # executing KL_SUM
    from abstractive.T5.T5 import T5
    dataset['t5'] = T5().multi_apply(original_texts, verbose=True)

    dataset.to_csv(output_path, index=False)
    print('file written to ' + os.path.abspath(output_path))

# run_all('../datasets/reviews.csv', '../outputs/all.reviews.summaries.csv')
run_all('../datasets/news.csv', '../outputs/all.news.summaries.csv')