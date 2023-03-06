class DialogRelationClassifier:

    def load_data(self):
        assert NotImplementedError

    def build_model(self):
        assert NotImplementedError

    def train_model(self, train_sequences, train_labels, test_sequences, test_labels, acc_out_file,
                    loss_out_file, model_out_file):

       assert NotImplementedError

    def load_model(self, model_file):
        assert NotImplementedError

    def get_dialog_relation(self, turn1, turn2):
        assert NotImplementedError

