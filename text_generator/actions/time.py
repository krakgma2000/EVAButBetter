from datetime import datetime
from .action import Action


class Time(Action):
    def run(self,action_obj=None):
        return "The current time in Barcelona is " + datetime.now().strftime("%H:%M")