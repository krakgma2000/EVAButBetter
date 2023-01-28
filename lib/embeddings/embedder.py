class Embedder:

    def __init__(self):
        pass

    def encode(self, text):
        raise NotImplementedError

    def __sent_emb(self, text_emb):
        raise NotImplementedError
