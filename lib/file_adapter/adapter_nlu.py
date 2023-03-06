from lib.file_adapter.adapter import FileAdapter
from lib.parsing.parser_yml import ParserYML
import re

NLU_RE = re.compile(r'\[(.+?)\]\((.+?)\)')


class AdapterIntentNLU(FileAdapter):
    def __init__(self, parser=ParserYML()):
        super().__init__(parser)

    def convert(self, filename):
        raw_data = self.parser.parse(filename)

        intents = []
        cleaned_sentences = []
        for intent in raw_data['nlu']:
            for example in intent['examples'].split('\n'):
                example_str = example.lstrip('- ')
                text = NLU_RE.sub(r'\1', example_str)
                cleaned_sentences.append(text)
                intents.append(intent["intent"])
        return cleaned_sentences, intents

# x,y = AdapterIntentNLU().convert('/Users/macbook_pro/Documents/GitHub/ChatBot/rasa_pipeline/data/nlu.yml')
# print(x)
# print(y)
