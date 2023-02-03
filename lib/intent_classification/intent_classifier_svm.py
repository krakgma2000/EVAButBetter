from lib.intent_classification.intent_classifier import Classifier

# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
from sklearn import svm
from matplotlib import pyplot as plt
import collections
import pickle
import numpy as np
import os

Cs_range = np.round(np.logspace(-4, 5, 20), 4)
gammas_range = np.round(np.logspace(-4, 5, 20), 4)


class SVM(Classifier):
    def __init__(self, kernel='linear'):
        super().__init__()
        self.model = svm.SVC(kernel=kernel)
        self.kernel = kernel
        self.C = 0
        self.gamma = 0

    def train(self, input, label, per=0.2, C_range=Cs_range, gammas_range=gammas_range):
        self.svm_grid_search(input, label, per=per, Cs_wide_range=C_range, gammas_wide_range=gammas_range)
        # construct smaller search ranges around the optimal result from the first search
        Cs_precise_range = np.round(
            np.linspace(C_range[self.C - 1], C_range[self.C + 1], 10),
            4)
        gammas_precise_range = np.round(
            np.linspace(gammas_range[self.gamma - 1], gammas_range[self.gamma + 1], 10),
            4)

        # Apply grid search algorithm for our C and gamma range:
        self.svm_grid_search(input, label, per=per, Cs_wide_range=Cs_precise_range,
                             gammas_wide_range=gammas_precise_range)

        self.model = svm.SVC(probability=True, kernel=self.kernel, decision_function_shape='ovr',
                             C=Cs_precise_range[self.C],
                             random_state=42, gamma=gammas_precise_range[self.gamma])
        self.model.fit(input, label)
        print("The model trained with the results from grid search has a accuracy of", self.model.score(input, label),
              "on the whole dataset.")
        self.save('svm_model')

    def save(self, filename):
        path = os.path.join(os.path.dirname(os.getcwd()), os.path.join("models", f'{filename}_{self.kernel}.pkl'))
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print("model saved at ", path)

    def load(self, filename):
        path = os.path.join(os.path.dirname(os.getcwd()), os.path.join("models", f'{filename}_{self.kernel}.pkl'))
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print("model loaded from", path)

    def predict(self, embedding):
        label = self.model.predict(embedding)
        prob = self.model.predict_proba(embedding)[0][label]
        return {'label': label.tolist()[0], 'prob': prob.tolist()[0]}

    def svm_grid_search(self, input, label, per, Cs_wide_range, gammas_wide_range):
        # split data
        given_random_state = 42
        x_train, x_test, y_train, y_test = train_test_split(input, label, test_size=per,
                                                            random_state=given_random_state)
        # compute model (grid seach with cross-validation to find best C and gamma)
        parameters = {"C": Cs_wide_range, "gamma": gammas_wide_range}

        # check cv size and lowest frequency of input labels
        frequency_dict = collections.Counter(label)
        min_size = min(freq for key, freq in frequency_dict.items())
        print(min_size)

        cv = 5
        if min_size < 5 and min_size > 1:
            cv = min_size
        clf = GridSearchCV(self.model, parameters, cv=min_size)
        print(y_train)
        clf.fit(x_train, y_train)

        bets_C_gamma = clf.best_params_
        best_train_score = clf.best_score_
        best_test_score = clf.score(x_test, y_test)

        print("Best C and gamma:", bets_C_gamma)
        print("Best training score: ", best_train_score)
        print("Best test score: ", best_test_score)

        # construct a heatmap to show the correlation of C, γ, and the accuracy
        # get all scores
        all_scores = clf.cv_results_["mean_test_score"].reshape(len(Cs_wide_range), len(gammas_wide_range))

        # find the maximum mean score
        max_acc = np.amax(all_scores)
        # find the indices to translate the to the corresponding C and γ
        indices_for_max_acc = np.argwhere(all_scores == max_acc)
        self.C = indices_for_max_acc[0, 0]
        self.gamma = indices_for_max_acc[0, 1]

        # prepare image
        plt.figure()
        plt.imshow(all_scores)
        plt.xlabel("gamma")
        plt.ylabel("C")
        plt.title("mean accuracy for tight grid")

        # set limits and range
        plt.xticks(np.arange(len(gammas_wide_range)), gammas_wide_range, rotation=45)
        plt.yticks(np.arange(len(Cs_wide_range)), Cs_wide_range)

        # set colorbar
        plt.colorbar()
        plt.show()
