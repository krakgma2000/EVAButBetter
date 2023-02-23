from .action import Action


class Greeting(Action):
    def run(self, action_obj=None):
        return "Hello!"
