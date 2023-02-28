import nltk
from nltk.corpus.reader import switchboard
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from nltk.classify.naivebayes import NaiveBayesClassifier
import pandas as pd
import random

nltk.download('switchboard')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class DialogRelationClassifier:
    def __init__(self):
        self.test_data = None
        self.train_data = None
        self.named_entities = None
        self.pos_tags = None
        self.switchboard_corpus = None
        self.classifier = None

    def load_data_1(self):
        # Load the Switchboard corpus
        self.switchboard_reader = switchboard.SwitchboardCorpusReader(
            "C://Users//gmorenoa//AppData//roaming//nltk_data//corpora//switchboard")
        self.turns = list(self.switchboard_reader.turns())
        print("We have %d turns" % len(self.turns))

        self.load_train_set_1()

    def load_data_2(self):
        self.train_df = pd.read_csv(
            "ubuntu-ranking-dataset-creator-master/ubuntu-ranking-dataset-creator-master/src/train.csv").reset_index()
        self.train_data = []
        for index, row in self.train_df.iterrows():
            turns = row["Utterance"].split("__eou__")
            self.train_data.append((turns[0], turns[1], row["Label"]))

        self.test_df = pd.read_csv(
            "ubuntu-ranking-dataset-creator-master/ubuntu-ranking-dataset-creator-master/src/test.csv").reset_index()
        self.test_data = []
        for index, row in self.train_df.iterrows():
            turns = row["Utterance"].split("__eou__")
            self.test_data.append((turns[0], turns[1], row["Label"]))

    def are_related(self, id1, speaker1, id2, speaker2):
        return speaker1 != speaker2 and id1 == id2 - 1

    def load_train_set_1(self):
        data = []

        # Get the dialogue turns in the conversation
        discourses = list(self.switchboard_reader.discourses())

        # Loop over all conversations in the corpus
        for discourse in discourses:

            # Loop over all turns in the conversation
            for i in range(1, len(discourse)):
                # Get the current and preceding posts
                current_post = discourse[i]
                preceding_post = discourse[i - 1]

                # Determine whether the current and preceding posts are related or not
                if self.are_related(preceding_post.id, preceding_post.speaker, current_post.id, current_post.speaker):
                    label = 1
                else:
                    label = 0

                # Add the current and preceding posts and their label to the list of dialogue turns
                data.append((" ".join(list(preceding_post)), " ".join(current_post), label))

        unrelated_size = 0
        finish = False
        while not finish:
            # Loop over all conversations in the corpus
            for discourse in discourses:
                random.shuffle(discourse)
                # Loop over all turns in the conversation
                for i in range(1, len(discourse)):
                    # Get the current and preceding posts
                    current_post = discourse[i]
                    preceding_post = discourse[i - 1]

                    # Determine whether the current and preceding posts are related or not
                    if self.are_related(preceding_post.id, preceding_post.speaker, current_post.id,
                                        current_post.speaker):
                        continue
                    else:
                        label = 0
                        unrelated_size += 1
                    # Add the current and preceding posts and their label to the list of dialogue turns
                    data.append((" ".join(list(preceding_post)), " ".join(current_post), label))

                if len(data) / 2 < unrelated_size:
                    finish = True
                    break

        random.shuffle(data)

        # Split the dialogue turns into training and test sets
        train_size = int(0.8 * len(data))

        self.train_data = data[:train_size]
        self.test_data = data[train_size:]

    def extract_features(self, previous_turn, current_turn):
        # Tokenize the turn and previous_turn
        current_turn_tokens = set(word_tokenize(current_turn))
        current_turn_tokens = {x.lower() for x in current_turn_tokens}
        previous_turn_tokens = set(word_tokenize(previous_turn))
        previous_turn_tokens = {x.lower() for x in previous_turn_tokens}
        # Create a dictionary of features
        features = {}
        # Feature 1: Check if the turn contains any words from the previous_turn
        features['current_contains_overlap'] = len(current_turn_tokens.intersection(previous_turn_tokens)) > 0
        # Feature 2: Check if the turn starts with a question word
        features['previous_starts_with_question'] = previous_turn_tokens.intersection(
            {'who', 'what', 'when', 'where', 'why', 'how'}) != set()
        features['current_starts_with_question'] = current_turn_tokens.intersection(
            {'who', 'what', 'when', 'where', 'why', 'how'}) != set()
        current_turn_da = nltk.pos_tag(word_tokenize(current_turn))
        features['current_turn_da'] = current_turn_da[0][1] if current_turn_da else None
        # Feature 4: Extract the dialog act of the previous turn
        prev_turn_da = nltk.pos_tag(word_tokenize(previous_turn))
        features['prev_turn_da'] = prev_turn_da[0][1] if prev_turn_da else None
        return features

    def train(self):
        # Extract the features from the training and test sets
        self.train_features_labels = [
            (self.extract_features(self.train_data[i][0], self.train_data[i][1]), self.train_data[i][2]) for i in
            range(len(self.train_data))]
        self.test_features_labels = [
            (self.extract_features(self.test_data[i][0], self.test_data[i][1]), self.test_data[i][2]) for i in
            range(len(self.test_data))]

        # Train a naive Bayes classifier on the training data
        self.classifier = NaiveBayesClassifier.train(self.train_features_labels)

    def test(self):
        accuracy = 0
        for test_sample in self.test_features_labels:
            # Predict whether the new turns are related or unrelated to each other
            prediction = self.classifier.classify(featureset=test_sample[0])
            if prediction == test_sample[1]:
                accuracy += 1

        print("Accuracy", accuracy / len(self.test_features_labels))


dialog_classifier = DialogRelationClassifier()
dialog_classifier.load_data_2()
dialog_classifier.train()
dialog_classifier.test()
