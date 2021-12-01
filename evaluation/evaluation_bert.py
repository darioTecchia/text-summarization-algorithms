from bert_score import score

import pandas

import os

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

def run_bert(input_path, output_path):
    print('reading ' + os.path.abspath(input_path))
    dataset = pandas.read_csv(input_path).truncate(after=config.SUMMARIES_CHUNK - 1)

    human_summaries = dataset['human_summaries'].fillna(" ").to_list()

    results = dict()

    for algorithm in config.SUMMARIZATION_ALGORITHMS:
        print('evalutating ' + algorithm)
        summaries = dataset[algorithm].fillna(" ").to_list()
        P, R, F1 = score(summaries, human_summaries, lang="en", verbose=True)
        results[algorithm + '_precision'] = P
        results[algorithm + '_recall'] = R
        results[algorithm + '_F1'] = F1

    pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
    print('file written to ' + os.path.abspath(output_path + '.csv'))

# run_bert("../outputs/all.reviews.summaries.csv", "../outputs/evaluation.reviews.bert")
run_bert("../outputs/all.news.summaries.csv", "../outputs/evaluation.news.bert")