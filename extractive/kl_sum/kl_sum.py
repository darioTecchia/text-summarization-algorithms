# Import the summarizer
from sumy.summarizers.kl import KLSummarizer

# Creating the parser
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

class KLSum:
    kl_summarizer = None

    def __init__(self):
        # Instantiating the KLSummarizer
        self.kl_summarizer = KLSummarizer()

    def apply(self, original_text = "", sentences_count = 3):
        summary = ""
        parser = PlaintextParser.from_string(original_text, Tokenizer('english'))
        for sentence in self.kl_summarizer(parser.document, sentences_count):
            summary += str(sentence)
        return summary

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing kl_sum summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text, 3))
        return summaries