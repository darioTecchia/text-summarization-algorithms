import pandas

import os

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

from summarizer_factory import get_summarizer

def run(input_path, output_path, algorithms):
    print('Running: ')
    print(*algorithms, sep = ", ") 
    print('Reading dataset from ' + os.path.abspath(input_path))
    print('\n')

    dataset = pandas.read_csv(input_path).truncate(after=config.SUMMARIES_CHUNK - 1)

    original_texts = dataset['original_texts'].fillna(" ").to_list()

    for algorithm in algorithms:
        Summarizer = get_summarizer(algorithm)
        dataset[algorithm] = Summarizer().multi_apply(original_texts, verbose=True)

    dataset.to_csv(output_path, index=False)

    print('Summaries file written to ' + os.path.abspath(output_path))
    print('\n')

    return dataset

import config

run('../datasets/news.csv', '../outputs/extractive.news.summaries.csv', config.EXTRACTIVE_ALGORITHMS)
run('../datasets/news.csv', '../outputs/abstractive.news.summaries.csv', config.ABSTRACTIVE_ALGORITHMS)
run('../datasets/news.csv', '../outputs/all.news.summaries.csv', config.All_ALGORITHMS)