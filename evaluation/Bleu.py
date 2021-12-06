from nltk.translate.bleu_score import sentence_bleu
import pandas, os
from transformers import logging
from evaluation.metric import Metric

logging.set_verbosity_error()

class Bleu(Metric):

    def __init__(self, algorithms = [], chunk_size = 5):
        self.algorithms = algorithms
        self.chunk_size = chunk_size

    def run(self, input_path, output_path):
        print('Reading dataset from ' + os.path.abspath(input_path))
        dataset = pandas.read_csv(input_path, header=0).truncate(after=self.chunk_size)

        results = dict()

        for algorithm in self.algorithms:
            results[algorithm + '_bleu'] = []

        for row in dataset.to_dict('records'):
            human_summary = row['human_summaries']
            for algorithm in self.algorithms:
                print('Evalutating ' + algorithm)
                summary = row[algorithm]
                bleu_score = sentence_bleu([human_summary.split()], summary.split())
                results[algorithm + '_bleu'].append(bleu_score)

        pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
        print('File written to ' + os.path.abspath(output_path + '.csv'))
        print('######## END ########\n')