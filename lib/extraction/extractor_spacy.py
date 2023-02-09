from lib.extraction.extractor_wrapper import Extractor
from spacy.tokens import Doc
import spacy

# pip install -U spacy
# python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")


class ExtractorSpaCy(Extractor):
    def __init__(self):
        super().__init__()

    def extract(self, tokens, ner=True, pos=True, morph=True, lemma=True):
        output_list = []

        for sent_tokens in tokens:
            output = dict()
            doc = Doc(nlp.vocab, words=sent_tokens)
            doc = nlp(doc)

            for i, token in enumerate(doc):
                word = {"token": token.text}
                if ner:
                    word["ner"] = token.ent_type_

                if pos:
                    word["pos"] = token.pos_
                if lemma:
                    word["lemma"] = token.lemma_

                output[token] = word
            output_list.append({"entities": output})

        return output_list

# print(ExtractorSpaCy().extract([["hi","you","are","Google"],['My',"Name","is","Alex"]]))
