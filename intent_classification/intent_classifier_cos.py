from intent_classification.intent_classifier import Classifier
import numpy as np
from numpy.linalg import norm


class CosSim:
    def __init__(self, intents, threshold=0.7):
        self.intents = intents  # {'label':[embedding,embedding]}
        self.threshold = 0.7

    def __cos_dist(self, input_sent_emb, intent_emb):
        try:
            cosin = np.dot(input_sent_emb, intent_emb) / (norm(input_sent_emb, ord=2) * norm(intent_emb, ord=2))
        except:
            cosin = 0

        return cosin

    def predict(self, embedding):
        max_each_intent = []
        for i, sent_intents_emb in enumerate(self.intents.values()):
            max_in_intent = 0
            for sent_emb in sent_intents_emb:
                cos = self.__cos_dist(embedding, sent_emb)
                if cos > max_in_intent:
                    max_in_intent = cos
            max_each_intent.append(max_in_intent)
        return list(self.intents.keys())[max_each_intent.index(max(max_each_intent))]


class ClassifierCos(Classifier):
    def __init__(self, model: CosSim):
        super().__init__()
        self.model = model

    def predict(self, embedding):
        self.model.predict(embedding)
