from .introduction import Introduction
from .apologies import Apologies
from .action import Action


class UnavailableService(Action):
    def __init__(self):
        self.apologies = Apologies()
        self.introduction = Introduction()
    def run(self, action_obj):
        service = action_obj["service"]
        run_intro = action_obj["run_intro"]
        to_return = self.apologies.run()
        if service == "location":
            to_return += " I can't access to your current location, so I can't help you with your request."
        elif service == "no_internet":
            to_return += " My internet connection is very limited, so I can't help you with your request."
        else:
            to_return += " I can't handle " + service + " services / information."

        if run_intro:
            to_return += " "
            to_return += self.introduction.run()

        return to_return
