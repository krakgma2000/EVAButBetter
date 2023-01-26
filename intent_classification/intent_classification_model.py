import numpy as np
import json
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pad_sequences import pad_sequences_multi


class IntentClassificatorModel:
    def __init__(self, data_path):
        self.all_labels = None
        self.dataset_path = data_path
        self.tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    def load_data(self, test_size):
        # Loading json data
        with open(self.dataset_path) as file:
            data = json.loads(file.read())

        # Loading out-of-scope intent data
        val_oos = np.array(data['oos_val'])
        train_oos = np.array(data['oos_train'])
        test_oos = np.array(data['oos_test'])

        # Loading other intents data
        val_others = np.array(data['val'])
        train_others = np.array(data['train'])
        test_others = np.array(data['test'])

        # Merging out-of-scope and other intent data
        val = np.concatenate([val_oos, val_others])
        train = np.concatenate([train_oos, train_others])
        test = np.concatenate([test_oos, test_others])
        data = np.concatenate([train, test, val])
        data = data.T

        text = data[0]
        labels = data[1]

        self.all_labels = labels

        return text, labels

    def one_hot_encoder(self, labels):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(np.unique(self.all_labels))
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder.fit(integer_encoded)

        labels_encoded = label_encoder.transform(labels)
        labels_encoded = labels_encoded.reshape(len(labels_encoded), 1)
        return onehot_encoder.transform(labels_encoded)

    def pad_sequences(self,train_txt,test_txt):
        ls = []
        for c in train_txt:
            ls.append(len(c.split()))
        maxLen = int(np.percentile(ls, 98))
        train_sequences = self.tokenizer.encode_batch(train_txt)
        train_sequences = pad_sequences_multi(train_sequences, maxlen=maxLen, padding='post')

        test_sequences = self.tokenizer.encode_batch(test_txt)
        test_sequences = pad_sequences_multi(test_sequences, maxlen=maxLen, padding='post')

        return train_sequences,test_sequences

    def embed(self):
        assert True
