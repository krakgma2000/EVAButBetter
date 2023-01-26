from tools import Tools

class Builder:
    def __init__(self,tools: Tools):
        self.tools = tools

    def build(self, text):
        tokenized_text = self.tools.tokenizer.tokenize_sent(text)
        tokenized_sent = self.tools.tokenizer.tokenize(tokenized_text)
        preprocessed_text = [self.tools.preprocessor(tokens) for tokens in tokenized_sent]
