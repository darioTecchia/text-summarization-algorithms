# Importing the model
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

class BART:

    tokenizer = None
    model = None

    def __init__(self):
        # Loading the model and tokenizer for bart-large-cnn
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def apply(self, original_text = "", return_tensors = 'pt'):
        original_text = original_text[:1024]
        # Encoding the inputs and passing them to model.generate()
        inputs = self.tokenizer.batch_encode_plus([original_text], return_tensors = return_tensors)
        summary_ids = self.model.generate(inputs['input_ids'], early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing BART summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text, 'pt'))
        return summaries