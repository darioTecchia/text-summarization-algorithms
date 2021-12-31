# pylint: disable=C0103,C0301
import os, requests, bz2, pandas
from evaluation.metric import Metric
from evaluation.s3_utils import S3, load_embeddings

dirname = os.path.dirname(__file__)

if not os.path.exists(os.path.join(dirname, "../resources")):
    os.mkdir(os.path.join(dirname, "../resources"))
if not os.path.exists(os.path.join(dirname, "../resources/deps.words")):
    print("Downloading the embeddings; this may take a while")
    url = "http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
    r = requests.get(url)
    d = bz2.decompress(r.content)
    with open(os.path.join(dirname, "../resources/deps.words"), "wb") as outputf:
        outputf.write(d)

class S3Metric(Metric):
    def __init__(self, algorithms = [], chunk_size = 5, emb_path=os.path.join(dirname, '../resources/deps.words'), \
        model_folder=os.path.join(dirname, '../resources/models/en/'), n_workers=24, tokenize=True):
        self.word_embeddings = load_embeddings(emb_path)
        self.model_folder = model_folder
        self.n_workers = n_workers
        self.tokenize = tokenize
        self.algorithms = algorithms
        self.chunk_size = chunk_size

    def run(self, input_path, output_path):
        print('Reading dataset from ' + os.path.abspath(input_path))
        dataset = pandas.read_csv(input_path, header=0).truncate(after=self.chunk_size)

        results = dict()

        for algorithm in self.algorithms:
            results[algorithm + '_s3_pyr'] = []
            results[algorithm + '_s3_resp'] = []

        for row in dataset.to_dict('records'):
            human_summary = row['human_summaries']
            for algorithm in self.algorithms:
                print('Evalutating ' + algorithm)
                summary = row[algorithm]
                score = S3([human_summary], [summary], word_embs=self.word_embeddings, model_folder=self.model_folder, tokenize=self.tokenize)
                results[algorithm + '_s3_pyr'].append(score[0])
                results[algorithm + '_s3_resp'].append(score[1])

        pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
        print('File written to ' + os.path.abspath(output_path + '.csv'))
        print('######## END ########\n')