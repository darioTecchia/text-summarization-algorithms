# Importing package and summarizer
import gensim
from gensim.summarization import summarize

class TextRank:

    def apply(self, original_text="", ratio=0.1, word_count=30):
        # Summarization when both ratio & word count is given
        # In case both are mentioned, then the summarize function ignores the ratio. So, only the word_count parameter is taken.
        try:
            return summarize(original_text, ratio, word_count)
        finally:
            return ""

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing textrank summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text, 0.1, 30))
        return summaries