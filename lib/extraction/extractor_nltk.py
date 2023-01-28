import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from extractor_wrapper import Extractor
from nltk.tag import pos_tag
from nltk import WordNetLemmatizer, ne_chunk
from nltk.chunk import tree2conlltags

sentence = "Mark and John are working at Google."

class ExtractorNLTK(Extractor):
    def __init__(self):
        super().__init__()

    def extract(self,tokens, ner=True, pos=True, morph=True, lemma=True):
        output = dict()
        ner_and_pos = tree2conlltags(ne_chunk(pos_tag(tokens)))
        for i,token in enumerate(tokens):
            word = {"token": token}
            if ner:
                word["ner"] = ner_and_pos[i][2]

            if pos:
                word["pos"] = ner_and_pos[i][1]
            if lemma:
                word["lemma"] = WordNetLemmatizer().lemmatize(token)

            output[token] = word

        return {"sentence": output}

# print(ExtractorNLTK().extract(["hi","you","are","Google","Alice","professor","floor"]))

