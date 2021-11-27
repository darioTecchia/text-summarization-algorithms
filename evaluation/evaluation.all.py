from bert_score import score

from rouge import Rouge
rouge = Rouge()

import pandas

from transformers import logging
logging.set_verbosity_error()

dataset = pandas.read_csv("../outputs/all.csv").truncate(after=4999)

human_summaries = dataset['human_summaries'].fillna(" ").to_list()

import sys
sys.path.append('..')

## evalutate KL_SUM
## bert score
kl_sum_summaries = dataset['kl_sum'].fillna(" ").to_list()
P, R, F1 = score(kl_sum_summaries, human_summaries, lang="en", verbose=True)
print(P, R, F1)
print('kl_sum: ')
print(f"F1 score: {F1.mean():.3f}, Precision score: {P.mean():.3f}, Recall score: {R.mean():.3f}")

## rouge score
kl_sum_score = rouge.get_scores(hyps=kl_sum_summaries, refs=human_summaries)
print(kl_sum_score[0])