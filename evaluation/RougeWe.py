import os, requests, bz2, pandas
from evaluation.metric import Metric
from evaluation.rouge_we_utils import rouge_n_we, load_embeddings

class RougeWe(Metric):

    def __init__(self, algorithms=[], chunk_size = 5, n_gram=3, tokenize=True):
        self.algorithms = algorithms
        self.chunk_size = chunk_size
        self.dirname = os.path.self.dirname(__file__)
        if not os.path.exists(os.path.join(self.dirname, "../resources")):
            os.mkdir(os.path.join(self.dirname, "../resources"))
        if not os.path.exists(os.path.join(self.dirname, "../resources/deps.words")):
            print("Downloading the embeddings; this may take a while")
            url = "http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
            r = requests.get(url)
            d = bz2.decompress(r.content)
            with open(os.path.join(self.dirname, "../resources/deps.words"), "wb") as outputf:
                outputf.write(d)

        self.word_embeddings = load_embeddings(os.path.join(self.dirname, '../resources/deps.words'))

    def run(self, input_path, output_path):
        print('Reading dataset from ' + os.path.abspath(input_path))
        dataset = pandas.read_csv(input_path, header=0).truncate(after=self.chunk_size)

        results = dict()

        for algorithm in self.algorithms:
            results[algorithm + '_precision'] = []
            results[algorithm + '_recall'] = []
            results[algorithm + '_F1'] = []

        for row in dataset.to_dict('records'):
            human_summary = row['human_summaries']
            for algorithm in self.algorithms:
                print('Evalutating ' + algorithm)
                summary = row[algorithm]
                score = rouge_n_we([summary], [human_summary], self.word_embeddings, self.n_gram, \
                    return_all=True, tokenize=self.tokenize)
                results[algorithm + '_precision'].append(score[0])
                results[algorithm + '_recall'].append(score[1])
                results[algorithm + '_F1'].append(score[2])

        pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
        print('File written to ' + os.path.abspath(output_path + '.csv'))
        print('######## END ########\n')
