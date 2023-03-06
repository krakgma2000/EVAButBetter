import json
import importlib, inspect

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
        print(intent)
        action_obj = self.db[intent]
        return action_obj

    def run_action(self, action_obj):

        module_name = "text_generator.actions." + action_obj["action"]  # the name of the module to import
        module = importlib.import_module(module_name)  # import the module

        # Get a list of all the attributes defined in the module
        module_attributes = inspect.getmembers(module, inspect.isclass)

        for name, cls in module_attributes:
            if cls.__module__ == module_name:
                if name in ["OOV","UnavailableService"]:
                    instance = cls()
                else:
                    instance = cls()
                result = instance.run(action_obj)
                return result
        print("No class found in module")



    def run_action_by_intent(self, intent):
        if intent in ["1","2","3","4","5","6"]:  #TODO: REMOVE THIS
            return "Sorry, but UPF services are not currently available.", 1

        action_obj = self.get_action_by_intent(intent)
        return self.run_action(action_obj), action_obj["next_action"]

    def test(self):
        for intent in self.db.keys():
            print("Intent: " + intent + ":\n")
            if intent == "schedule_meeting":
                print("stop")
            self.run_action_by_intent(intent)
            print("\n************************\n\n")
