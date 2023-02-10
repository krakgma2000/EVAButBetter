from lib.extraction.extractor_wrapper import Extractor
from spacy.tokens import Doc
import spacy

# pip install -U spacy
# python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# sents =
print(nlp("Hello World. Flight at 12:00, how are you professor Bonada Sanjaume Joi Jiu Ko? Where is AI&ML lab?").ents)


# print([ee.label_ for ee in sents.ents])


class ExtractorSpaCy(Extractor):
    def __init__(self):
        super().__init__()

    def extract_sent(self, sent):
        if isinstance(sent, str):
            sents = nlp(sent)
            output_list = {"data": [{'value': e.text, 'entity': e.label_} for e in sents.ents]}
            return output_list
        elif isinstance(sent, list):
            output_list = [
                {"data": [{'value': e.text, 'entity': e.label_} for e in nlp(sentence).ents]} for
                sentence in sent]
            return output_list

    def extract_tokens(self, tokens, ner=True, pos=True, morph=True, lemma=True, skip_empty=True):
        output_list = []
        for sent_tokens in tokens:
            doc = Doc(nlp.vocab, words=sent_tokens)
            doc = nlp(doc)

            for i, token in enumerate(doc):
                word = {"value": token.text}
                if ner:
                    word["entity"] = token.ent_type_

                if pos:
                    word["pos"] = token.pos_
                if lemma:
                    word["lemma"] = token.lemma_

                if word["entity"] and skip_empty:
                    output_list.append({"data": word})

        return output_list

# print(ExtractorSpaCy().extract([["hi","you","are","Google"],['My',"Name","is","Alex"]]))
