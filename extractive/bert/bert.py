# Import the summarizer
from summarizer import Summarizer

class BERT:
    model = None

    def __init__(self):
        # Instantiating the KLSummarizer
        self.model = Summarizer()

    def apply(self, original_text = ""):
        return self.model(original_text)

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing BERT summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text))
        return summaries