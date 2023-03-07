import sys
from intent_classification.intent_classification_model import IntentClassificationModel

argv = sys.argv
domain_dataset_path = None
if len(argv) >= 2:
    domain_dataset_path = argv[1]

# **** BUILD GENERIC INTENT CLASSIFIER****#
generic_dataset_path = "./intent_classification/datasets/generic.yml"
embedder_train_dataset_path = "./intent_classification/datasets/glove.6B.100d.txt"

print("#### TRAINING GENERIC INTENT CLASSIFIER ####\n")
intentClassModel = IntentClassificationModel(embedder_train_dataset_path, domain_dataset_path)  #TODO: ADAPT CODE TO MAKE .YML AS INPUT DATASET

text,labels = intentClassModel.load_data_generic(generic_dataset_path)

label_encoder_output="./intent_classification/utils/generic_label_encoder.pkl"
tokenizer_output = "./intent_classification/utils/generic_tokenizer.pkl"
model_output = "./intent_classification/models/generic_intent_classifier.h5"
accuracy_output = "./intent_classification/plots/generic_accuracy.png"
loss_output = "./intent_classification/plots/generic_loss.png"

intentClassModel.execute_train_pipeline(text, labels, label_encoder_output, tokenizer_output, accuracy_output,
                                        loss_output, model_output)

print("#### VALIDATING GENERIC INTENT CLASSIFIER ON DOMAIN / NOT-DOMAIN CLASSIFICATION ####\n")
intentClassModel.validate_generic_model(generic_dataset_path)


# **** BUILD DOMAIN INTENT CLASSIFIER ****#
print("#### TRAINING DOMAIN INTENT CLASSIFIER ####\n")

text,labels = intentClassModel.load_data_domain()   #TODO: ADAPT CODE TO MAKE .YML AS INPUT DATASET
label_encoder_output="./intent_classification/utils/domain_label_encoder.pkl"
tokenizer_output = "./intent_classification/utils/domain_tokenizer.pkl"
model_output = "./intent_classification/models/domain_intent_classifier.h5"
accuracy_output = "./intent_classification/plots/domain_accuracy.png"
loss_output = "./intent_classification/plots/domain_loss.png"
intentClassModel.execute_train_pipeline(text, labels, label_encoder_output, tokenizer_output, accuracy_output,
                                        loss_output, model_output)

print("#### FINISHED BUILDING INTENT CLASSIFIERS ####")

#TODO: OTHER CLASSIFIERS