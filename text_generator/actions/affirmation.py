from .action import Action


class Affirmation(Action):
    def run(self, action_obj=None):
        return "Yes."
