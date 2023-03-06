import yaml
import io
from lib.parsing.parser import Parser


class ParserYML(Parser):
    def __init__(self):
        data = dict()
        super().__init__(data)

    def parse(self, filename):
        with open(filename, 'r', encoding='utf8') as stream:
            self.data = yaml.safe_load(stream)
        return self.data

    def to_file(self, data, filename):
        with io.open(filename, 'w', encoding='utf8') as outfile:
            yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


# print(ParserYML().parse('/Users/macbook_pro/Documents/GitHub/EVAButBetter/lib/parsing/rules.yml'))
