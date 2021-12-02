from nltk.translate.bleu_score import sentence_bleu

import pandas

import os

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

def run_bleu(input_path, output_path):
    print('Reading dataset from ' + os.path.abspath(input_path))
    dataset = pandas.read_csv(input_path, header=0).truncate(after=config.SUMMARIES_CHUNK - 1)

    results = dict()

    for algorithm in config.All_ALGORITHMS:
        results[algorithm + '_bleu'] = []

    for row in dataset.to_dict('records'):
        human_summary = row['human_summaries']
        for algorithm in config.All_ALGORITHMS:
            print('Evalutating ' + algorithm)
            summary = row[algorithm]
            bleu_score = sentence_bleu([human_summary.split()], summary.split())
            results[algorithm + '_bleu'].append(bleu_score)

    pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
    print('File written to ' + os.path.abspath(output_path + '.csv'))
    print('######## END ########\n')