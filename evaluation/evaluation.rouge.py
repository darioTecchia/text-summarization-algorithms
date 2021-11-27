from rouge import Rouge
rouge = Rouge()

with open('../text.txt', 'r') as file:
    data = file.read().replace('\n', '')

from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('..')

def print_rouge_score(score):
    print(f"ROUGE-1: F1 score: {score[0]['rouge-1']['f']:.3f}, Precision score: {score[0]['rouge-1']['p']:.3f}, Recall score: {score[0]['rouge-1']['r']:.3f}")
    print(f"ROUGE-2: F1 score: {score[0]['rouge-2']['f']:.3f}, Precision score: {score[0]['rouge-2']['p']:.3f}, Recall score: {score[0]['rouge-2']['r']:.3f}")
    print(f"ROUGE-L: F1 score: {score[0]['rouge-l']['f']:.3f}, Precision score: {score[0]['rouge-l']['p']:.3f}, Recall score: {score[0]['rouge-l']['r']:.3f}")

# evalutate KL_SUM
from extractive.kl_sum.kl_sum import KLSum
summary = KLSum().apply(data)
kl_sum_score = rouge.get_scores(summary, data)
print('kl_sum: ')
print_rouge_score(kl_sum_score)

# evalutate lexrank
from extractive.lexrank.lexrank import LexRank
summary = LexRank().apply(data)
lexrank_score = rouge.get_scores(summary, data)
print('lexrank: ')
print_rouge_score(lexrank_score)

# evalutate lsa
from extractive.lsa.lsa import LSA
summary = LSA().apply(data)
lsa_score = rouge.get_scores(summary, data)
print('lsa: ')
print_rouge_score(lsa_score)

# evalutate luhn
from extractive.luhn.luhn import Luhn
summary = Luhn().apply(data)
luhn_score = rouge.get_scores(summary, data)
print('luhn: ')
print_rouge_score(luhn_score)

# evalutate textrank
from extractive.textrank.textrank import TextRank
summary = TextRank().apply(data)
textrank_score = rouge.get_scores(summary, data)
print('textrank: ')
print_rouge_score(textrank_score)

# evalutate bert
from extractive.bert.bert import BERT
summary = BERT().apply(data)
textrank_score = rouge.get_scores(summary, data)
print('bert: ')
print_rouge_score(textrank_score)

# evalutate T5
from abstractive.T5.T5 import T5
summary = T5().apply(data)
t5_score = rouge.get_scores(summary, data)
print('T5: ')
print_rouge_score(t5_score)

# evalutate BART
from abstractive.BART.BART import BART
summary = BART().apply(data)
bart_score = rouge.get_scores(summary, data)
print('BART: ')
print_rouge_score(bart_score)