import pandas, os, numpy
from transformers import logging
from evaluation.metric import Metric

logging.set_verbosity_error()

dirname = os.path.dirname(__file__)

class MoverScore(Metric):

    def __init__(self, algorithms = [], chunk_size = 5, version=2, \
                stop_wordsf=os.path.join(dirname, '../resources/stopwords.txt'), \
                n_gram=1, remove_subwords=True, batch_size=48):
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

    def run(self, input_path, output_path):
        print('Reading dataset from ' + os.path.abspath(input_path))
        dataset = pandas.read_csv(input_path).truncate(after=self.chunk_size)

        results = dict()

        human_summaries = dataset['human_summaries'].fillna(" ").to_list()

        for algorithm in self.algorithms:

            print('Evalutating ' + algorithm)
            summaries = dataset[algorithm].fillna(" ").to_list()

            refs = human_summaries
            if isinstance(human_summaries[0], list):
                refs = [" ".join(ref) for ref in human_summaries]
                
            idf_dict_summ = self.get_idf_dict(summaries)
            idf_dict_ref = self.get_idf_dict(refs)
            scores = []
            if isinstance(human_summaries[0], list):
                for reference, summary in zip(human_summaries, summaries):
                    s = self.word_mover_score(reference, [summary]*len(reference), idf_dict_ref, idf_dict_summ, \
                            stop_words=self.stop_words, n_gram=self.n_gram, remove_subwords=self.remove_subwords,\
                            batch_size=self.batch_size)
                    scores.append(numpy.mean(s))
            else:
                scores = self.word_mover_score(human_summaries, summaries, idf_dict_ref, idf_dict_summ, \
                                stop_words=self.stop_words, n_gram=self.n_gram, remove_subwords=self.remove_subwords,\
                                batch_size=self.batch_size)

            
            score_dict = [{"mover_score" : score} for score in scores]
            print(score_dict)

        pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
        print('File written to ' + os.path.abspath(output_path + '.csv'))
        print('######## END ########\n')