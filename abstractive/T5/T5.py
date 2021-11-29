# Importing requirements
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

class T5:
    my_model = None
    tokenizer = None

    def __init__(self):
        # Instantiating the model and tokenizer
        self.my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def apply(self, original_text = "", return_tensors = 'pt', max_length = 600):
        text = "summarize:" + original_text
        input_ids = self.tokenizer.encode(text, return_tensors = return_tensors, max_length = max_length, truncation=True)
        summary_ids = self.my_model.generate(input_ids)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def multi_apply(self, original_texts = [], verbose = False):
        summaries = []
        for index, original_text in enumerate(original_texts):
            if(verbose):
                print("Computing T5 summaries {:2.2f}".format(float(index) * 100.0 / float(len(original_texts))) + " %")
            summaries.append(self.apply(original_text))
        return summaries