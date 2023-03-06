class Embedder:

    def __init__(self, model_emb):
        self.model_emb = model_emb

    def encode(self, text):
        raise NotImplementedError

    def __sent_emb(self, text_emb):
        raise NotImplementedError
