import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from matplotlib import pyplot as plt
from tensorflow import keras
import pickle
import pandas as pd

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

    # **** TRAIN ****#S
    def load_data_generic(self, generic_dataset_path):
        # Loading json data
        with open(generic_dataset_path) as file:
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

        domain_texts, domain_labels = self.load_data_domain(replace_original_labels=True)

        text = np.concatenate([text, domain_texts])
        labels = np.concatenate([labels, domain_labels])

        self.intents = np.unique(labels)

        return text, labels

    def load_data_domain(self, replace_original_labels=True):
        df = pd.read_csv(self.domain_dataset_path)
        if replace_original_labels:
            df["label"] = "upf"
        texts = df["sentence"].to_numpy()
        labels = df["label"].to_numpy()

        self.intents = np.unique(labels)

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

    def get_intent(self, text):

        text = [text]
        test_keras = self.tokenizer.texts_to_sequences(text)
        input_size = self.model.input.type_spec.shape[1]
        test_keras_sequence = keras.preprocessing.sequence.pad_sequences(test_keras, maxlen=input_size,
                                                                         padding='post')
        pred = self.model.predict(test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(pred, 1))[0]


# **** BUILD EXAMPLE (GENERIC)****#
generic_dataset_path = "./datasets/data_full.json"
embedder_train_dataset_path = "./datasets/glove.6B.100d.txt"
domain_dataset_path = "./datasets/train_set.csv"
intentClassModel = IntentClassificationModel(embedder_train_dataset_path, domain_dataset_path)
text,labels = intentClassModel.load_data_generic(generic_dataset_path)
label_encoder_output="./utils/generic_label_encoder.pkl"
tokenizer_output = "./utils/generic_tokenizer.pkl"
model_output = "./models/generic_intent_classifier.h5"
accuracy_output = "./plots/generic_accuracy.png"
loss_output = "./plots/generic_loss.png"
intentClassModel.execute_train_pipeline(text, labels, label_encoder_output, tokenizer_output, accuracy_output,
                                        loss_output, model_output)

# **** BUILD EXAMPLE (DOMAIN) ****#
text,labels = intentClassModel.load_data_domain(replace_original_labels=False)
label_encoder_output="./utils/domain_label_encoder.pkl"
tokenizer_output = "./utils/domain_tokenizer.pkl"
model_output = "./models/domain_intent_classifier.h5"
accuracy_output = "./plots/domain_accuracy.png"
loss_output = "./plots/domain_loss.png"
intentClassModel.execute_train_pipeline(text, labels, label_encoder_output, tokenizer_output, accuracy_output,
                                        loss_output, model_output)

# **** PREDICT EXAMPLE ****#
intentClassModel = IntentClassificationModel(embedder_train_dataset_path,domain_dataset_path)
intentClassModel.load_model(model_file="./models/generic_intent_classifier.h5")
intentClassModel.load_tokenizer(tokenizer_file="./utils/generic_tokenizer.pkl")
intentClassModel.load_label_encoder(label_encoder_file="./utils/generic_label_encoder.pkl")
print(intentClassModel.get_intent("Thank you"))

def validate_model(type):
    embedder_train_dataset_path = "./datasets/glove.6B.100d.txt"
    domain_dataset_path = "./datasets/validation_set.csv"
    intentClassModel = IntentClassificationModel(embedder_train_dataset_path, domain_dataset_path)

    intentClassModel.load_model(model_file="./models/" + type + "_intent_classifier.h5")
    intentClassModel.load_tokenizer(tokenizer_file="./utils/" + type + "_tokenizer.pkl")
    intentClassModel.load_label_encoder(label_encoder_file="./utils/" + type + "_label_encoder.pkl")
    if type == "domain":
        replace_original_labels = False
    else:
        replace_original_labels = True
    texts, labels = intentClassModel.load_data_domain(replace_original_labels=replace_original_labels)

    score = 0
    for i, text in enumerate(texts):
        intent = intentClassModel.get_intent(text)
        if intent == labels[i]:
            score += 1
            print(intent,labels[i])
        else:
            print(text, intent, labels[i])
    print("The " + type + " model accuracy is "+'{:.1%}'.format(score / len(texts)))


validate_model("generic")
validate_model("domain")
