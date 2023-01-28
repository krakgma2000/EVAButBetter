from tools import Tools

class Builder:
    def __init__(self,tools: Tools):
        self.tools = tools
    def build(self, text):
        tokenized_text = self.tools.tokenizer.tokenize_sent(text)
        tokenized_sent = self.tools.tokenizer.tokenize(tokenized_text)
        preprocessed_text = [self.tools.preprocessor.preprocess(tokens) for tokens in tokenized_sent]
        extracted_data = self.tools.extractor.extract(preprocessed_text)
        sentence_embedding = self.tools.embedder.encode(text)
        return(extracted_data)


# print(Builder(Tools()).build("Hello World. how are yo?"))