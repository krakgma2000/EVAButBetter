from lib.intent_classification.intent_classifier import Classifier


class CosSim:
    def __init__(self, intents):
        self.intents = intents
        pass

    def predict(self):
        pass


class ClassifierCos(Classifier):
    def __init__(self, model: CosSim):
        super().__init__(model)
        self.model = model
        pass

    def predict(self, embeddings):
        pass
