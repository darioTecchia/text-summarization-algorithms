from bert_score import score
import pandas, os
from transformers import logging
from evaluation.metric import Metric

logging.set_verbosity_error()

class Bert(Metric):

    def __init__(self, algorithms = [], chunk_size = 5):
        self.algorithms = algorithms
        self.chunk_size = chunk_size

    def run(self, input_path, output_path):
        print('Reading dataset from ' + os.path.abspath(input_path))
        dataset = pandas.read_csv(input_path).truncate(after=self.chunk_size)

        human_summaries = dataset['human_summaries'].fillna(" ").to_list()

        results = dict()

        for algorithm in self.algorithms:
            print('Evalutating ' + algorithm)
            summaries = dataset[algorithm].fillna(" ").to_list()
            P, R, F1 = score(summaries, human_summaries, lang="en", verbose=True, model_type='bert-base-uncased')
            results[algorithm + '_precision'] = P
            results[algorithm + '_recall'] = R
            results[algorithm + '_F1'] = F1

        pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
        print('File written to ' + os.path.abspath(output_path + '.csv'))
        print('######## END ########\n')