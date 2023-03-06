from lib.extraction.extractor_wrapper import Extractor
from lib.parsing.parser import Parser
from lib.parsing.parser_yml import ParserYML
from spacy.tokens import Doc
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import random
from pathlib import Path
from tqdm import tqdm
import os
import re

# pip install -U spacy
# python -m spacy download en_core_web_sm


MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()),
                          os.path.join("models", 'ner'))
# print(MODEL_PATH)
ENTITY_RE = re.compile(r'\[(.+?)\]\((.+?)\)')


class ExtractorSpaCy(Extractor):
    def __init__(self, model_name=MODEL_PATH):
        super().__init__()

        try:
            self.model = spacy.load(model_name)
        except:
            self.model = spacy.load('en_core_web_sm')

    def extract_sent(self, sent):
        if isinstance(sent, str):
            sents = self.model(sent)
            output_list = {"data": [{'value': e.text, 'entity': e.label_} for e in sents.ents]}
            return output_list
        elif isinstance(sent, list):
            output_list = [
                {"data": [{'entity': ent.label_, 'start': ent.start_char, 'end': ent.end_char, 'value': ent.text,
                           'extractor': 'SpaCy'} for ent in
                          self.model(sentence).ents]} for
                sentence in sent]
            return output_list

    def extract_tokens(self, tokens, ner=True, pos=True, morph=True, lemma=True, skip_empty=True):
        output_list = []
        for sent_tokens in tokens:
            doc = Doc(nlp.vocab, words=sent_tokens)
            doc = self.model(doc)

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

    def convert_rasa_to_spacy(self, nlu_file, parser: Parser = ParserYML()):
        data = parser.parse(nlu_file)
        sentences = []
        for intent in data['nlu']:
            for example in intent['examples'].split('\n'):
                example_str = example.lstrip('- ')
                text = ENTITY_RE.sub(r'\1', example_str)  # Remove entity labels and brackets
                entities = []
                for match in ENTITY_RE.finditer(example_str):
                    entity_text = match.group(1)
                    entity_label = match.group(2)
                    start_index = match.start(1)
                    end_index = match.end(1)
                    entities.append((start_index - 1,
                                     end_index - 1,
                                     # "value": entity_text,
                                     entity_label
                                     ))
                training_data = (text, {"entities": entities})
                if (entities != []):
                    sentences.append(training_data)
        return sentences

    def train_ner(self, output_dir, config_path, n_iter=100):
        self.model = spacy.blank('en')
        print("Created blank 'en' model")
        if 'ner' not in self.model.pipe_names:
            ner = self.model.create_pipe('ner')
            self.model.add_pipe('ner', last=True)
        else:
            ner = self.model.get_pipe('ner')
        data = self.convert_rasa_to_spacy(config_path)
        for _, annotations in data:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != 'ner']
        with self.model.disable_pipes(*other_pipes):  # only train NER
            self.model.begin_training()
            for itn in range(n_iter):
                random.shuffle(data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))
                for batch in tqdm(batches):
                    for text, annotations in batch:
                        # create Example
                        doc = self.model.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        # Update the model
                        self.model.update([example], losses=losses, drop=0.3)
                print("Losses", losses)

        # Save model
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            self.model.to_disk(output_dir)
            print("Saved model to", output_dir)


# ExtractorSpaCy().train_ner('/Users/macbook_pro/Documents/GitHub/EVAButBetter/models/ner',
#                            '/Users/macbook_pro/Documents/GitHub/EVAButBetter/rasa_pipeline/data/nlu.yml')
# print(ExtractorSpaCy().convert_rasa_to_spacy(
#     '/Users/macbook_pro/Documents/GitHub/EVAButBetter/rasa_pipeline/data/nlu.yml'))
