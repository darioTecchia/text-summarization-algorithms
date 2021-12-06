from rouge import Rouge

from evaluation.metric import Metric

import pandas, os

from transformers import logging
logging.set_verbosity_error()

class RougeWe(Metric):

    def __init__(self, algorithms=[], chunk_size = 5):
        self.algorithms = algorithms
        self.chunk_size = chunk_size
        self.rouge = Rouge()

    def run(self, input_path, output_path):
        print('Reading dataset from ' + os.path.abspath(input_path))

        dataset = pandas.read_csv(input_path).truncate(after=self.chunk_size)

        human_summaries = dataset['human_summaries'].fillna(" ").to_list()

        results_1 = dict()
        results_2 = dict()
        results_l = dict()

        for algorithm in self.algorithms:
            print('Evalutating ' + algorithm)
            summaries = dataset[algorithm].fillna(" ").to_list()

            rouge_score = self.rouge.get_scores(summaries, human_summaries)

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
