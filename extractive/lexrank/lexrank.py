# Importing the parser and tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# Import the LexRank summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer

class LexRank:

    lex_rank_summarizer = None

    def __init__(self):
        self.lex_rank_summarizer = LexRankSummarizer()

    def apply(self, original_text="", sentences_count=3):
        # Initializing the parser
        my_parser = PlaintextParser.from_string(original_text, Tokenizer('english'))

        # Creating a summary of 3 sentences.
        summary = ""
        for sentence in self.lex_rank_summarizer(my_parser.document, sentences_count):
            summary += str(sentence)
        return summary

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing lexrank summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text, 3))
        return summaries