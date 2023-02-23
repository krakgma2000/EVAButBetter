from .action import Action


class YourWelcome(Action):
    def run(self, action_obj=None):
        return "Your welcome!"
