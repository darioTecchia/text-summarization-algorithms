import pandas, os, numpy
from transformers import logging
from evaluation.metric import Metric

from collections import defaultdict

logging.set_verbosity_error()

dirname = os.path.dirname(__file__)

class MoverScore(Metric):

    def __init__(self, algorithms = [], chunk_size = 5, version=2, \
                stop_wordsf=os.path.join(dirname, '../resources/stopwords.txt'), \
                n_gram=1, remove_subwords=True, batch_size=48, aggregate=True):
        self.algorithms = algorithms
        self.chunk_size = chunk_size
        self.version = version
        if self.version == 1:
            from moverscore import get_idf_dict, word_mover_score
        else:
            from moverscore_v2 import get_idf_dict, word_mover_score
        self.get_idf_dict = get_idf_dict
        self.word_mover_score = word_mover_score
        stop_words = []
        if stop_wordsf is not None:
            with open(stop_wordsf) as inputf:
                stop_words = inputf.read().strip().split(' ')
        self.stop_words = stop_words
        self.n_gram = n_gram
        self.remove_subwords = remove_subwords
        self.batch_size = batch_size
        self.idf_dict_ref = defaultdict(lambda: 1.)
        self.idf_dict_hyp = defaultdict(lambda: 1.)

    def run(self, input_path, output_path):
        print('Reading dataset from ' + os.path.abspath(input_path))
        dataset = pandas.read_csv(input_path, header=0).truncate(after=self.chunk_size)

        results = dict()

        for algorithm in self.algorithms:
            results[algorithm + '_bleu'] = []

        for row in dataset.to_dict('records'):
            human_summary = row['human_summaries'][:512]
            for algorithm in self.algorithms:
                print('Evalutating ' + algorithm)
                summary = row[algorithm][:512]
                score = self.word_mover_score([human_summary], [summary], self.idf_dict_ref, self.idf_dict_hyp, self.stop_words,
                                        n_gram=1, remove_subwords=True)
                results[algorithm + '_bleu'].append(score[0])

        pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
        print('File written to ' + os.path.abspath(output_path + '.csv'))
        print('######## END ########\n')