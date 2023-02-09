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

# import time
# tools = Tools()
# from lib.extraction.extractor_spacy import ExtractorSpaCy
# tools.extractor = ExtractorSpaCy()
# tmp = time.time()
# print(Builder(tools).build("Hello World. Flight at 12:00, how are you professor Jorge?"))
# print(time.time()-tmp)
# from lib.extraction.extractor_nltk import ExtractorNLTK
# tools.extractor = ExtractorNLTK()
# tmp = time.time()
# print(Builder(tools).build("Hello World. how are yo?"))
# print(time.time()-tmp)