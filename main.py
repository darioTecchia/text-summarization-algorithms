from runners.runner import run
import config

# run('datasets/news.csv', 'outputs/all.news.summaries.csv', config.All_ALGORITHMS)
# run('datasets/reviews.csv', 'outputs/all.reviews.summaries.csv', config.All_ALGORITHMS)

from evaluation.evaluation_bert import run_bert
from evaluation.evaluation_bleu import run_bleu
from evaluation.evaluation_rouge import run_rouge
from evaluation.evaluation_rouge_we import run_rouge_we

# run_bert("outputs/all.news.summaries.csv", "outputs/evaluation.news.bert")
# run_bert("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.bert")

# run_bleu("outputs/all.news.summaries.csv", "outputs/evaluation.news.bleu")
# run_bleu("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.bleu")

# run_rouge("outputs/all.news.summaries.csv", "outputs/evaluation.news.rouge")
# run_rouge("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.rouge")

run_rouge_we("outputs/all.news.summaries.csv", "outputs/evaluation.news.rouge_we")
run_rouge_we("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.rouge_we")