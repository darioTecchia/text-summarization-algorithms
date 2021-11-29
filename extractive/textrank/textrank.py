# Importing package and summarizer
import gensim
from gensim.summarization import summarize

class TextRank:

    def apply(self, original_text = "", word_count = 120):
        return summarize(original_text, word_count = word_count)

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing textrank summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text))
        return summaries