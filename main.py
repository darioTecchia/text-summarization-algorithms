from runners.runner import run
import config

if __name__ == '__main__':
    # run('datasets/news.csv', 'outputs/all.news.summaries.csv', config.All_ALGORITHMS)
    # run('datasets/reviews.csv', 'outputs/all.reviews.summaries.csv', config.All_ALGORITHMS)

    # from evaluation.Bert import Bert
    # bert = Bert(config.All_ALGORITHMS, config.SUMMARIES_CHUNK)

    # bert.run("outputs/all.news.summaries.csv", "outputs/evaluation.news.bert")
    # bert.run("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.bert")

    # from evaluation.Bleu import Bleu
    # bleu = Bleu(config.All_ALGORITHMS, config.SUMMARIES_CHUNK)

    # bleu.run("outputs/all.news.summaries.csv", "outputs/evaluation.news.bleu")
    # bleu.run("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.bleu")

    # from evaluation.Rouge import Rouge
    # rouge = Rouge(config.All_ALGORITHMS, config.SUMMARIES_CHUNK)

    # rouge.run("outputs/all.news.summaries.csv", "outputs/evaluation.news.rouge")
    # rouge.run("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.rouge")

    # from evaluation.RougeWe import RougeWe
    # rouge_we = RougeWe(config.All_ALGORITHMS, config.SUMMARIES_CHUNK)

    # rouge_we.run("outputs/all.news.summaries.csv", "outputs/evaluation.news.rouge_we")
    # rouge_we.run("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.rouge_we")

    from evaluation.MoverScore import MoverScore
    mover_score = MoverScore(config.All_ALGORITHMS, config.SUMMARIES_CHUNK, version=1)

    mover_score.run("outputs/all.reviews.summaries.csv", "outputs/evaluation.reviews.mover_score")
    mover_score.run("outputs/all.news.summaries.csv", "outputs/evaluation.news.mover_score")
    