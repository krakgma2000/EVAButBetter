import json
import importlib


class IntentToAction:
    def __init__(self, actions_db_dir):
        self.actions_db_dir = actions_db_dir
        self.db = None
        self.loadDB()

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
        module_attributes = dir(module)

        # Find the name of the class in the module
        class_name = module_attributes[-9-retry]

        if class_name is not None:
            class_ = getattr(module, class_name)
            try:
                instance = class_()
            except:
                self.run_action(action_obj,retry=retry+1)
                return
            result = instance.run(action_obj)
            print("Action Result: " + result)
            return result
        else:
            print("No class found in module")



    def run_action_by_intent(self, intent):
        action_obj = self.get_action_by_intent(intent)
        self.run_action(action_obj)

    def test(self):
        for intent in self.db.keys():
            print("Intent: " + intent + ":\n")
            self.run_action_by_intent(intent)
            print("\n************************\n\n")


subject = IntentToAction("intent_to_action.json")
subject.test()

