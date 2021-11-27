from bert_score import score

import pandas

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

dataset = pandas.read_csv("../outputs/all.summaries.csv").truncate(after=config.SUMMARIES_CHUNK - 1)

human_summaries = dataset['human_summaries'].fillna(" ").to_list()

results = dict()

## evalutate BERT
bert_summaries = dataset['bert'].fillna(" ").to_list()
## bert score
P, R, F1 = score(bert_summaries, human_summaries, lang="en", verbose=True)
results['bert_precision'] = P
results['bert_recall'] = R
results['bert_F1'] = F1

## evalutate KL_SUM
kl_sum_summaries = dataset['kl_sum'].fillna(" ").to_list()
## bert score
P, R, F1 = score(kl_sum_summaries, human_summaries, lang="en", verbose=True)
results['kl_sum_precision'] = P
results['kl_sum_recall'] = R
results['kl_sum_F1'] = F1

## evalutate lexrank
lexrank_summaries = dataset['lexrank'].fillna(" ").to_list()
## bert score
P, R, F1 = score(lexrank_summaries, human_summaries, lang="en", verbose=True)
results['lexrank_precision'] = P
results['lexrank_recall'] = R
results['lexrank_F1'] = F1

## evalutate lsa
lsa_summaries = dataset['lsa'].fillna(" ").to_list()
## bert score
P, R, F1 = score(lsa_summaries, human_summaries, lang="en", verbose=True)
results['lsa_precision'] = P
results['lsa_recall'] = R
results['lsa_F1'] = F1

## evalutate luhn
luhn_summaries = dataset['luhn'].fillna(" ").to_list()
## bert score
P, R, F1 = score(luhn_summaries, human_summaries, lang="en", verbose=True)
results['luhn_precision'] = P
results['luhn_recall'] = R
results['luhn_F1'] = F1

## evalutate textrank
# textrank_summaries = dataset['textrank'].fillna(" ").to_list()
## bert score
# P, R, F1 = score(textrank_summaries, human_summaries, lang="en", verbose=True)
# results['textrank_precision'] = P
# results['textrank_recall'] = R
# results['textrank_F1'] = F1

## evalutate bart
bart_summaries = dataset['bart'].fillna(" ").to_list()
## bert score
P, R, F1 = score(bart_summaries, human_summaries, lang="en", verbose=True)
results['bart_precision'] = P
results['bart_recall'] = R
results['bart_F1'] = F1

pandas.DataFrame(results).to_csv('../outputs/evaluation.bert.csv', index=False)