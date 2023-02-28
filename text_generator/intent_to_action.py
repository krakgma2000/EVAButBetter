import json
import importlib, inspect

from transformers import T5Tokenizer, T5ForConditionalGeneration

from actions.action import Action

class IntentToAction:
    def __init__(self, actions_db_dir):
        self.actions_db_dir = actions_db_dir
        self.db = None
        self.loadDB()
        self.load_t5()

    def load_t5(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base')

    def loadDB(self):
        db = json.load(open(self.actions_db_dir))
        self.db = db
        return db

    def get_action_by_intent(self, intent):
        action_obj = self.db[intent]
        return action_obj

    def run_action(self, action_obj,retry=0):

        module_name = "actions." + action_obj["action"]  # the name of the module to import
        module = importlib.import_module(module_name)  # import the module

        # Get a list of all the attributes defined in the module
        module_attributes = inspect.getmembers(module, inspect.isclass)

        for name, cls in module_attributes:
            if cls.__module__ == module_name:
                if name in ["OOV","UnavailableService"]:
                    instance = cls(tokenizer=self.tokenizer,t5=self.t5)
                else:
                    instance = cls()
                result = instance.run(action_obj)
                print("Action Result: " + result)
                return result
        print("No class found in module")



    def run_action_by_intent(self, intent):
        action_obj = self.get_action_by_intent(intent)
        self.run_action(action_obj)

    def test(self):
        for intent in self.db.keys():
            print("Intent: " + intent + ":\n")
            if intent == "schedule_meeting":
                print("stop")
            self.run_action_by_intent(intent)
            print("\n************************\n\n")


subject = IntentToAction("intent_to_action.json")
subject.test()

