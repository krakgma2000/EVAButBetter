import random
from .action import Action


class Randomizer(Action):
    def run(self,action_obj):
        type = action_obj["type"]
        if type == "coin":
            flip = random.randint(0, 1)
            if flip == 0:
                result = "Heads!"
            else:
                result = "Tails!"
            return f"Lets flip a coin. *flip*. The result is... {result}."
        elif type == "dice":
            result = random.randint(1,6)
            return f"Lets roll a dice. *Rolling*. The result is... {result}."