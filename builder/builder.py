from tools import Tools

class Builder:
    def __init__(self,tools: Tools):
        self.tools = tools
    def build(self, text):
        tokenized_text = self.tools.tokenizer.tokenize_sent(text)
        tokenized_sent = self.tools.tokenizer.tokenize(tokenized_text)
        preprocessed_text = [self.tools.preprocessor.reconstruct(self.tools.preprocessor.preprocess(tokens)) for tokens in tokenized_sent]
        extracted_sentence_data = self.tools.extractor.extract_sent(preprocessed_text)
        print(extracted_sentence_data)
        sentence_embeddings = self.tools.embedder.encode(preprocessed_text)
        intents = [self.tools.dialog_act_classifier.predict(emb.get('embedding')) for emb in sentence_embeddings]
        return(intents)


# print(Builder(Tools()).build("Hello World. how are yo?"))

import time
tmp = time.time()
tools = Tools()
from lib.extraction.extractor_spacy import ExtractorSpaCy
tools.extractor = ExtractorSpaCy()

print(Builder(tools).build("Hello World World World. Flight at 12:00, how are you professor Bonada Sanjaume Jordi?"))
print(time.time()-tmp)
# from lib.extraction.extractor_nltk import ExtractorNLTK
# tools.extractor = ExtractorNLTK()
# tmp = time.time()
# print(Builder(tools).build("Hello World. how are yo?"))
# print(time.time()-tmp)


# list1 = [{"A": 1, "B": 2}, {"C": 3, "D": 4}]
# list2 = [{"E": 5, "F": 6}, {"G": 7, "H": 8}]
#
# result = []
# for i in range(len(list1)):
#     result.append({**list1[i], **list2[i]})
#
# print(result)