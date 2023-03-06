import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))


class Preprocessor:

    def __init__(self, stop_words=None):
        self.stop_words = stop_words

    def preprocess(self, tokens, to_lower=False):
        output = []
        for w in tokens:
            if to_lower:
                w = w.lower()
            if self.stop_words is not None:
                if w in self.stop_words:
                    continue
            output.append(w)

        return output

    def reconstruct(self, tokens):
        return " ".join(tokens)

# print(Preprocessor(stop_words=stops).preprocess(["hi","you","?"]))
