from sentence_transformers import SentenceTransformer
import numpy as np
from lib.embeddings.embedder import Embedder

MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
model = SentenceTransformer(MODEL_NAME)


class EmbedderBERT(Embedder):
    def __init__(self, model_emb=model):
        super().__init__(model_emb)

    def encode(self, text):
        result = []
        if isinstance(text, str):
            result = {"embedding": {text: self.__sent_emb(text)}}
        elif isinstance(text, list):
            result = [{"embedding": self.__sent_emb(sentence)} for sentence in text]
        return result

    def __sent_emb(self, text):
        return self.model_emb.encode(text)
