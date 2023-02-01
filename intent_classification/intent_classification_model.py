import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from matplotlib import pyplot as plt
from tensorflow import keras
import pickle

'''
loss: 0.1976 - acc: 0.9490 - val_loss: 0.5196 - val_acc: 0.9042
train time: 35m
'''

class IntentClassificationModel:
    def __init__(self, train_data_path, embedder_train_data_path):
        self.train_data_path = train_data_path
        self.embedder_train_data_path = embedder_train_data_path

        self.model = None

        self.label_encoder = None
        self.word_index = None
        self.all_labels = None
        self.tokenizer = keras.preprocessing.text.Tokenizer()
        self.intents = None
        self.embedding_matrix = None

    # **** TRAIN ****#S
    def load_data(self):
        # Loading json data
        with open(self.train_data_path) as file:
            data = json.loads(file.read())

        # out-of-scope intent data
        val_oos = np.array(data['oos_val'])
        train_oos = np.array(data['oos_train'])
        test_oos = np.array(data['oos_test'])

        # other intents data
        val_others = np.array(data['val'])
        train_others = np.array(data['train'])
        test_others = np.array(data['test'])

        # Joining
        val = np.concatenate([val_oos, val_others])
        train = np.concatenate([train_oos, train_others])
        test = np.concatenate([test_oos, test_others])
        data = np.concatenate([train, test, val])
        data = data.T

        text = data[0]
        labels = data[1]

        self.intents = np.unique(labels)

        return text, labels

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

    def execute_train_pipeline(self):
        text, labels = self.load_data()
        encoded_labels = self.one_hot_encoder(labels, output_file="./utils/label_encoder.pkl")
        train_txt, test_txt, train_labels, test_labels = train_test_split(text, encoded_labels, test_size=0.1)

        self.build_tokenizer(train_txt, output_file="./utils/tokenizer.pkl")
        self.build_embedding_matrix()

        train_sequences, test_sequences = self.raw_text_to_sequences(train_txt, test_txt)

        self.build_model(input_length=train_sequences.shape[1])

        self.train_model(train_sequences, train_labels, test_sequences, test_labels,
                         acc_out_file="./plots/accuracy.png", loss_out_file="./plots/loss.png",
                         model_out_file="./models/generic_intent_classifier.h5")

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

    def get_intent(self, text):

        text = [text]
        test_keras = self.tokenizer.texts_to_sequences(text)
        test_keras_sequence = keras.preprocessing.sequence.pad_sequences(test_keras, maxlen=16,
                                                                         padding='post')
        pred = self.model.predict(test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(pred, 1))[0]


# **** BUILD ****#
train_dataset_path = "./datasets/data_full.json"
embedder_train_dataset_path = "./datasets/glove.6B.100d.txt"
intentClassModel = IntentClassificationModel(train_dataset_path, embedder_train_dataset_path)
intentClassModel.execute_train_pipeline()

# **** PREDICT ****#
intentClassModel.load_model(model_file="./models/generic_intent_classifier.h5")
intentClassModel.load_tokenizer(tokenizer_file="./utils/tokenizer.pkl")
intentClassModel.load_label_encoder(label_encoder_file="./utils/label_encoder.pkl")
intent = intentClassModel.get_intent("See you later")
print(intent)
