import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from matplotlib import pyplot as plt
from tensorflow import keras
import pickle
import pandas as pd
from lib.file_adapter.adapter_nlu import AdapterIntentNLU

'''
- 107s 312ms/step - loss: 0.1703 - acc: 0.9529 - val_loss: 0.4475 - val_acc: 0.9085
train time: 35m
The generic model accuracy is 98.5%

4s 387ms/step - loss: 0.0230 - acc: 0.9967 - val_loss: 0.0567 - val_acc: 0.9851
train time: 2m
The domain model accuracy is 96.4%
'''


class IntentClassificationModel:
    def __init__(self, embedder_train_data_path, domain_dataset_path):

        self.embedder_train_data_path = embedder_train_data_path
        self.domain_dataset_path = domain_dataset_path

        self.model = None

        self.label_encoder = None
        self.word_index = None
        self.all_labels = None
        self.tokenizer = keras.preprocessing.text.Tokenizer()
        self.intents = None
        self.embedding_matrix = None
        self.adapter = AdapterIntentNLU()

    # **** TRAIN ****#S
    def load_data_generic(self, generic_dataset_path):

        text,labels = self.adapter.convert(generic_dataset_path)

        domain_texts, domain_labels = self.load_data_domain(replace_original_labels=True)

        text = np.concatenate([text, domain_texts])
        labels = np.concatenate([labels, domain_labels])

        self.intents = np.unique(labels)

        return text, labels

    def load_data_domain(self, replace_original_labels=False):

        texts, labels = self.adapter.convert(self.domain_dataset_path)
        if replace_original_labels:
            labels = np.full(len(labels),"upf")
        return texts, labels

    def one_hot_encoder(self, labels, output_file):
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(self.intents)

        with open(output_file, 'wb') as file:
            pickle.dump(self.label_encoder, file)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder.fit(integer_encoded)

        labels_encoded = self.label_encoder.transform(labels)
        labels_encoded = labels_encoded.reshape(len(labels_encoded), 1)
        return onehot_encoder.transform(labels_encoded)

    def build_tokenizer(self, train_txt, output_file):
        self.tokenizer.fit_on_texts(train_txt)
        self.word_index = self.tokenizer.word_index

        with open(output_file, 'wb') as file:
            pickle.dump(self.tokenizer, file)

    def build_embedding_matrix(self):
        embeddings_index = {}
        with open(self.embedder_train_data_path, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_dim = len(embeddings_index['the'])
        n_words = len(self.word_index) + 1
        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (n_words, embedding_dim))

        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def raw_text_to_sequences(self, train_txt, test_txt):
        ls = []
        for c in train_txt:
            ls.append(len(c.split()))

        maxLen = int(np.percentile(ls, 98))

        train_sequences = self.tokenizer.texts_to_sequences(train_txt)
        train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=maxLen, padding='post')

        test_sequences = self.tokenizer.texts_to_sequences(test_txt)
        test_sequences = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=maxLen, padding='post')

        return train_sequences, test_sequences

    def build_model(self, input_length):
        model = keras.models.Sequential()

        vocab_length = len(self.word_index) + 1
        model.add(
            keras.layers.Embedding(vocab_length, 100, trainable=False, input_length=input_length,
                                   weights=[self.embedding_matrix]))
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(len(self.intents), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        self.model = model

    def train_model(self, train_sequences, train_labels, test_sequences, test_labels, acc_out_file,
                    loss_out_file, model_out_file):

        history = self.model.fit(train_sequences, train_labels, epochs=20,
                                 batch_size=64, shuffle=True,
                                 validation_data=[test_sequences, test_labels])

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(acc_out_file)

        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(loss_out_file)

        self.model.save(model_out_file)

    def execute_train_pipeline(self, text, labels, label_encoder_output, tokenizer_output, accuracy_output, loss_output,
                               model_output):

        encoded_labels = self.one_hot_encoder(labels, output_file=label_encoder_output)
        train_txt, test_txt, train_labels, test_labels = train_test_split(text, encoded_labels, test_size=0.1)

        self.build_tokenizer(train_txt, output_file=tokenizer_output)
        self.build_embedding_matrix()

        train_sequences, test_sequences = self.raw_text_to_sequences(train_txt, test_txt)

        self.build_model(input_length=train_sequences.shape[1])

        self.train_model(train_sequences, train_labels, test_sequences, test_labels,
                         acc_out_file=accuracy_output, loss_out_file=loss_output,
                         model_out_file=model_output)

    # **** PREDICT ****#
    def load_model(self, model_file):
        self.model = keras.models.load_model(model_file)

    def load_tokenizer(self, tokenizer_file):
        with open(tokenizer_file, 'rb') as file:
            self.tokenizer = pickle.load(file)
            self.word_index = self.tokenizer.word_index

    def load_label_encoder(self, label_encoder_file):
        with open(label_encoder_file, 'rb') as file:
            self.label_encoder = pickle.load(file)

    def validate_generic_model(self, generic_dataset_path):
        self.load_model(model_file="./intent_classification/models/generic_intent_classifier.h5")
        self.load_tokenizer(tokenizer_file="./intent_classification/utils/generic_tokenizer.pkl")
        self.load_label_encoder(label_encoder_file="./intent_classification/utils/generic_label_encoder.pkl")
        texts, labels = self.load_data_generic(generic_dataset_path)
        _, texts, _, labels = train_test_split(texts, labels, test_size=0.1)
        upf_score = 0
        others_score = 0
        total_upf = 0
        total_others = 0
        for i, text in enumerate(texts[:1000]):
            intent_obj = self.get_intent(text)
            intent = intent_obj["intent"]
            if labels[i] == "upf":
                total_upf += 1
                if intent == "upf":
                    upf_score += 1
            else:
                total_others += 1
                if intent != "upf":
                    others_score += 1

        print("The generic model predicts upf intents with a " + '{:.1%}'.format(upf_score / total_upf))
        print("The generic model predicts non-upf intents with a " + '{:.1%}'.format(others_score / total_others))

    def get_intent(self, text):

        text = [text]
        test_keras = self.tokenizer.texts_to_sequences(text)
        input_size = self.model.input.type_spec.shape[1]
        test_keras_sequence = keras.preprocessing.sequence.pad_sequences(test_keras, maxlen=input_size,
                                                                         padding='post')
        pred = self.model.predict(test_keras_sequence)

        intent_object = {"intent":str(self.label_encoder.inverse_transform(np.argmax(pred, 1))[0]), "prob":np.argmax(pred, 1)}

        return intent_object
