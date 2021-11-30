import pandas

import os

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

def run_abstractive(input_path, output_path):
    print('reading ' + os.path.abspath(input_path))

    dataset = pandas.read_csv(input_path).truncate(after=config.SUMMARIES_CHUNK - 1)

    original_texts = dataset['original_texts'].fillna(" ").to_list()

    # executing BART
    from abstractive.BART.BART import BART
    dataset['bart'] = BART().multi_apply(original_texts, verbose=True)

    # executing KL_SUM
    from abstractive.T5.T5 import T5
    dataset['kl_sum'] = T5().multi_apply(original_texts, verbose=True)

    dataset.to_csv(output_path, index=False)
    print('file written to ' + os.path.abspath(output_path))

run_abstractive('../datasets/reviews.csv', '../outputs/extractive.reviews.summaries.csv')

run_abstractive('../datasets/news.csv', '../outputs/extractive.news.summaries.csv')