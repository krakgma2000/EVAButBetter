import nltk
nltk.download("stopwords")
from nltk.tokenize import sent_tokenize, word_tokenize


class Tokenizer:
    def __init__(self, tokenizer=word_tokenize):
        self.tokenizer = tokenizer

    def tokenize_sent(self, text):
        return sent_tokenize(text)

    def tokenize(self, text):
        #if input is list of chunked text
        if isinstance(text, list):
            return [self.tokenizer(line) for line in text]

        #if text is str:
        return self.tokenizer(text)

# print(Tokenizer().tokenize(["I like eat meal. I appreciate","I like to fly"]))
# print(Tokenizer().tokenize("I like eat meal. I like to fly"))
# print(Tokenizer().tokenize(["I like eat meal. I like to fly"]))
