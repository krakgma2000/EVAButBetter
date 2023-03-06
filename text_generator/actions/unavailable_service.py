from .introduction import Introduction
from .apologies import Apologies
from .action import Action


class UnavailableService(Action):
    def __init__(self):
        self.apologies = Apologies()
        self.introduction = Introduction()

    def concatenate_strings(self, strings):
        return ' '.join(strings)

    def run(self, action_obj):
        service = action_obj["service"]
        run_intro = action_obj["run_intro"]
        apologies = self.apologies.run()
        standard_sentence = "I can't help you with your request."
        if service == "location":
            to_return = self.concatenate_strings([apologies, "I can't access to your current location",standard_sentence])
        elif service == "no_internet":
            to_return = self.concatenate_strings([apologies, "My internet connection is very limited",standard_sentence])
        else:
            to_return = self.concatenate_strings([apologies, "I can't handle " + service + " services or information",standard_sentence])

        if run_intro:
            to_return += " "
            to_return += self.introduction.run()

        return to_return
