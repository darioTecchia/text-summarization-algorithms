from nltk.translate.bleu_score import sentence_bleu
from numpy import array

import pandas

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

dataset = pandas.read_csv("../outputs/all.summaries.csv", header=0).truncate(after=config.SUMMARIES_CHUNK - 1)

human_summaries = dataset['human_summaries'].fillna(" ").to_list()

results = dict()

for algorithm in config.SUMMARIZATION_ALGORITHMS:
    results[algorithm + '_bleu'] = []

for row in dataset.to_dict('records'):
    human_summary = row['original_texts']
    for algorithm in config.SUMMARIZATION_ALGORITHMS:
        print('evalutating ' + algorithm)
        summary = row[algorithm]
        bleu_score = sentence_bleu(human_summaries, summary)
        results[algorithm + '_bleu'].append(bleu_score)

print(results)

pandas.DataFrame(results).to_csv('../outputs/evaluation.bleu.csv', index=False)