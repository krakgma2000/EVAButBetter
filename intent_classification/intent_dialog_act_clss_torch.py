from intent_classification.intent_classifier import Classifier
from neural_networks.sentence_bert_classifier import SentenceBERTClassifier

model_sentence_bert = SentenceBERTClassifier()
class ClassifierDA(Classifier):
    def __init__(self, model = model_sentence_bert):
        super().__init__()
        self.model = model

    def predict(self, embedding):
        self.model.predict(embedding)