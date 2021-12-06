import os
import requests
import bz2
from evaluation.rouge_we_utils import rouge_n_we, load_embeddings
import pandas

import sys
sys.path.append('..')

dirname = os.path.dirname(__file__)

import config

def run_rouge_we(input_path, output_path, n_gram=3, tokenize=True):
    print('Reading dataset from ' + os.path.abspath(input_path))
    dataset = pandas.read_csv(input_path, header=0).truncate(after=config.SUMMARIES_CHUNK - 1)

    if not os.path.exists(os.path.join(dirname, "embeddings")):
        os.mkdir(os.path.join(dirname, "embeddings"))
    if not os.path.exists(os.path.join(dirname, "embeddings/deps.words")):
        print("Downloading the embeddings; this may take a while")
        url = "http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
        r = requests.get(url)
        d = bz2.decompress(r.content)
        with open(os.path.join(dirname, "embeddings/deps.words"), "wb") as outputf:
            outputf.write(d)

    word_embeddings = load_embeddings(os.path.join(dirname, './embeddings/deps.words'))

    results = dict()

    for algorithm in config.All_ALGORITHMS:
        results[algorithm + '_precision'] = []
        results[algorithm + '_recall'] = []
        results[algorithm + '_F1'] = []

    for row in dataset.to_dict('records'):
        human_summary = row['human_summaries']
        for algorithm in config.All_ALGORITHMS:
            print('Evalutating ' + algorithm)
            summary = row[algorithm]
            score = rouge_n_we([summary], [human_summary], word_embeddings, n_gram, \
                 return_all=True, tokenize=tokenize)
            results[algorithm + '_precision'].append(score[0])
            results[algorithm + '_recall'].append(score[1])
            results[algorithm + '_F1'].append(score[2])

    pandas.DataFrame(results).to_csv(output_path + '.csv', index=False)
    print('File written to ' + os.path.abspath(output_path + '.csv'))
    print('######## END ########\n')