from rouge import Rouge
rouge = Rouge()

import pandas

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

dataset = pandas.read_csv("../outputs/all.summaries.csv").truncate(after=config.SUMMARIES_CHUNK - 1)

human_summaries = dataset['human_summaries'].fillna(" ").to_list()

results = dict()

## evalutate KL_SUM
kl_sum_summaries = dataset['kl_sum'].fillna(" ").to_list()
## rouge score
kl_sum_scores = rouge.get_scores(hyps=kl_sum_summaries, refs=human_summaries)
print(kl_sum_scores)