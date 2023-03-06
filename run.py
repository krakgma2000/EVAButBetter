from speech_recognition.recorder import Recorder
from speech_recognition.speech_recognition_whisper import SpeechRecognitionWhisper
from intent_classification.intent_classification_model import IntentClassificationModel
from text_generator.intent_to_action import IntentToAction
import sys

argv = sys.argv

print("#### STARTING EVA... BUT BETTER! ####\n")

recorder = Recorder()

speech_recognition_system = SpeechRecognitionWhisper()

speech_recognition_system.print_description()

generic_dataset_path = "./datasets/data_full.json"
embedder_train_dataset_path = "./datasets/glove.6B.100d.txt"
intent_classification_generic = IntentClassificationModel(embedder_train_dataset_path, generic_dataset_path)
intent_classification_generic.load_model(model_file="./intent_classification/models/generic_intent_classifier.h5")
intent_classification_generic.load_tokenizer(tokenizer_file="./intent_classification/utils/generic_tokenizer.pkl")
intent_classification_generic.load_label_encoder(
    label_encoder_file="./intent_classification/utils/generic_label_encoder.pkl")

intent_classification_domain = IntentClassificationModel(embedder_train_dataset_path, generic_dataset_path)
intent_classification_domain.load_model(model_file="./intent_classification/models/domain_intent_classifier.h5")
intent_classification_domain.load_tokenizer(tokenizer_file="./intent_classification/utils/domain_tokenizer.pkl")
intent_classification_domain.load_label_encoder(
    label_encoder_file="./intent_classification/utils/domain_label_encoder.pkl")

actions_db_dir = "./text_generator/intent_to_action.json"
intent_to_action = IntentToAction(actions_db_dir)

if (len(argv) >= 2 and argv[1] == "text"):

    ### TEXT INPUT / OUTPUT ###
    print("You have chosen text input, so please type your request (in English):\n")

    end = False
    while not end:
        eng_transcription = input('Type here:\n')

        intent_obj = intent_classification_generic.get_intent(eng_transcription)

        intent_name = intent_obj["intent"]

        if intent_name == "upf":
            intent_obj = intent_classification_domain.get_intent(eng_transcription)
            intent_name = intent_obj["intent"]

        answer, next_action = intent_to_action.run_action_by_intent(intent_name)

        print(answer)
        print("\n")

        if next_action == 0:  # Action 0: end
            end = True
        elif next_action == 1:  # Action 1: start again
            continue
        elif next_action == 2:  # Action 2: slot filling
            assert NotImplementedError #TODO

else:

    ### LISTENING ###
    print("You have chosen audio input/output, so please make your request now to your microphone\n")

    end = False
    while not end:
        recording_path = recorder.listen()

        lang, eng_transcription = speech_recognition_system.run(recording_path)

        intent_obj = intent_classification_generic.get_intent(eng_transcription)

        intent_name = intent_obj["intent"]
        if intent_name == "upf":
            intent_obj = intent_classification_domain.get_intent(eng_transcription)
            intent_name = intent_obj["intent"]

        answer, next_action = intent_to_action.run_action_by_intent(intent_name)

        print(answer) #TODO: REPLACE WITH TTS
        print("\n")

        if next_action == 0:  # Action 0: end
            end = True
        elif next_action == 1:  # Action 1: start again
            continue
        elif next_action == 2:  # Action 2: slot filling
            assert NotImplementedError #TODO
