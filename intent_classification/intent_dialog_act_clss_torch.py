from intent_classification.intent_classifier import Classifier
from neural_networks.sentence_bert_classifier import SentenceBERTClassifier

model_sentence_bert = SentenceBERTClassifier()


class ClassifierDA(Classifier):
    def __init__(self, model=model_sentence_bert):
        super().__init__()
        self.model = model

    def predict(self, embedding):
        return {'dialog_act': self.model.predict(embedding)}

# from sentence_transformers import SentenceTransformer
#
# MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
# model_bert = SentenceTransformer(MODEL_NAME)
#
# import time
#
# encoding = model_bert.encode("Where are my pants?")
# time_start = time.time()
# print(ClassifierDA().predict(encoding))
# print(time.time() - time_start)
