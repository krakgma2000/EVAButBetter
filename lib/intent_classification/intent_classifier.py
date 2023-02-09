class Classifier:
    def __init__(self, model):
        self.model = model

    def predict(self, embeddings):
        raise NotImplementedError
