import torch
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()),
                          os.path.join("models", 'mrda_fnn_312_1000ep.pth'))
print(MODEL_PATH)


class MLP(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        result = self.layer1(x)
        result = self.relu(result)
        result = self.softmax(result)
        return result


class SentenceBERTClassifier:
    def __init__(self, file_name=MODEL_PATH):
        data = torch.load(file_name)

        self.input_size = data["input_size"]
        self.output_size = data["output_size"]
        self.model_state = data["model_state"]
        self.ids = data["ids"]

        self.model = MLP(self.input_size, self.output_size)
        self.model.load_state_dict(self.model_state)

    def predict(self, embedding, return_type=None):
        with torch.no_grad():
            proba = self.model(torch.tensor(embedding))

            if return_type == 'label':
                return self.ids[proba.argmax()]
            elif return_type == 'index':
                return proba.argmax()
            elif return_type == 'proba':
                max(proba).item()
            return {'label': self.ids[proba.argmax()], 'prob': max(proba).item()}

# from sentence_transformers import SentenceTransformer
#
# MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
# model_bert = SentenceTransformer(MODEL_NAME)
#
# import time
#
# encoding = model_bert.encode("Where are my pants?")
# time_start = time.time()
# print(SentenceBERTClassifier().predict(encoding))
# print(time.time() - time_start)
