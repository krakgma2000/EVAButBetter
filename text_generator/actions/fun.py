import pyjokes, randfacts
from .action import Action


class Fun(Action):
    def run(self,action_obj):
        type = action_obj["type"]
        if type == 'fact':
            return pyjokes.get_joke()
        elif type == "joke":
            return randfacts.get_fact()
