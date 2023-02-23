from datetime import date
from .action import Action


class Date(Action):
    def run(self, action_obj=None):
        return "The current date in Barcelona is " + date.today().strftime("%d/%m/%Y")