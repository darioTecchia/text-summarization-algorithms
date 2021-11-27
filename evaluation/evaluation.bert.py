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

for algorithm in config.SUMMARIZATION_ALGORITHMS:
    print('evalutating ' + algorithm)
    summaries = dataset[algorithm].fillna(" ").to_list()
    P, R, F1 = score(summaries, human_summaries, lang="en", verbose=True)
    results[algorithm + '_precision'] = P
    results[algorithm + '_recall'] = R
    results[algorithm + '_F1'] = F1

pandas.DataFrame(results).to_csv('../outputs/evaluation.bert.csv', index=False)