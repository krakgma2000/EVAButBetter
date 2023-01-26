import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from extractor_wrapper import Extractor
from nltk.tag import pos_tag
from nltk import WordNetLemmatizer, ne_chunk

class ExtractorNLTK(Extractor):
    def __init__(self):
        super().__init__()

    def extract(self,tokens, ner=True, pos=True, morph=True, lemma=True):
        output = dict()
        # for token in tokens:
        #     word = {"token": token}
        #     if ner:
        #         word["ner"] = ne_chunk(pos_tag(tokens))
        #     if pos:
        #         word["pos"] = pos_tag(tokens)
        #     if lemma:
        #         word["lemma"] = WordNetLemmatizer().lemmatize(token)
        #
        #     output[token] = word
        #
        # return {"sentence": output}

# print(ExtractorNLTK().extract(["hi","you","are","Google"]))

