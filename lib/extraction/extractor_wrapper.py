class Extractor:

    def __init__(self):
        pass

    def extract_sent(self, sent):
        raise NotImplementedError

    def extract_tokens(self, tokens, ner=True, pos=True, morph=True, lemma=True, skip_empty=True):
        raise NotImplementedError
