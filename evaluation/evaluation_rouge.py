from rouge import Rouge
rouge = Rouge()

import pandas

import os

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

import config

def run_rouge(input_path, output_path):
    print('Reading dataset from ' + os.path.abspath(input_path))

    dataset = pandas.read_csv(input_path).truncate(after=config.SUMMARIES_CHUNK - 1)

    human_summaries = dataset['human_summaries'].fillna(" ").to_list()

    results_1 = dict()
    results_2 = dict()
    results_l = dict()

    for algorithm in config.SUMMARIZATION_ALGORITHMS:
        print('Evalutating ' + algorithm)
        summaries = dataset[algorithm].fillna(" ").to_list()

        rouge_score = rouge.get_scores(summaries, human_summaries)

        results_1[algorithm + '_precision'] = [score['rouge-1']['p'] for score in rouge_score]
        results_1[algorithm + '_recall'] = [score['rouge-1']['r'] for score in rouge_score]
        results_1[algorithm + '_F1'] = [score['rouge-1']['f'] for score in rouge_score]

        results_2[algorithm + '_precision'] = [score['rouge-2']['p'] for score in rouge_score]
        results_2[algorithm + '_recall'] = [score['rouge-2']['r'] for score in rouge_score]
        results_2[algorithm + '_F1'] = [score['rouge-2']['f'] for score in rouge_score]

        results_l[algorithm + '_precision'] = [score['rouge-l']['p'] for score in rouge_score]
        results_l[algorithm + '_recall'] = [score['rouge-l']['r'] for score in rouge_score]
        results_l[algorithm + '_F1'] = [score['rouge-l']['f'] for score in rouge_score]

    pandas.DataFrame(results_1).to_csv(output_path + '.rouge_1.csv', index=False)
    pandas.DataFrame(results_2).to_csv(output_path + '.rouge_2.csv', index=False)
    pandas.DataFrame(results_l).to_csv(output_path + '.rouge_l.csv', index=False)

    print('File written to ' + os.path.abspath(output_path + '.rouge_1.csv'))
    print('File written to ' + os.path.abspath(output_path + '.rouge_2.csv'))
    print('File written to ' + os.path.abspath(output_path + '.rouge_l.csv'))
    print('######## END ########\n')

# run_rouge("../outputs/all.reviews.summaries.csv", "../outputs/evaluation.reviews.rouge")
run_rouge("../outputs/all.news.summaries.csv", "../outputs/evaluation.news.rouge")