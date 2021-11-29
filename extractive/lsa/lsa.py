# Import the summarizer
from sumy.summarizers.lsa import LsaSummarizer

# Parsing the text string using PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

class LSA:
    lsa_summarizer = None

    def __init__(self):
        self.lsa_summarizer = LsaSummarizer()

    def apply(self, original_text="", sentences_count = 4):
        summary = ""
        parser = PlaintextParser.from_string(original_text, Tokenizer('english'))
        for sentence in self.lsa_summarizer(parser.document, sentences_count = sentences_count):
            summary += str(sentence)
        return summary

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing lsa summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text))
        return summaries