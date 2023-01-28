from lib.preprocessing.preprocessor import Preprocessor
from lib.preprocessing.tokenizer import Tokenizer

class Tools():
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.preprocessor = Preprocessor()
        self.extractor = None #implement NER and etc