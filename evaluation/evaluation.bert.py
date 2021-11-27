from bert_score import score

import pandas

from transformers import logging
logging.set_verbosity_error()

dataset = pandas.read_csv("../datasets/all.csv").truncate(after=4999)

human_summaries = dataset['human_summaries'].to_list()

import sys
sys.path.append('..')

## evalutate KL_SUM
from extractive.kl_sum.kl_sum import KLSum
kl_sum_summaries = dataset['kl_sum'].to_list()
P, R, F1 = score(kl_sum_summaries, human_summaries, lang="en", verbose=True)
print(P, R, F1)
# print('kl_sum: ')
# print(f"F1 score: {F1.mean():.3f}, Precision score: {P.mean():.3f}, Recall score: {R.mean():.3f}")