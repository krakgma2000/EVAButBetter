from .action import Action


class Goodbye(Action):
    def run(self,action_obj=None):
        return "Bye! I hope to see you soon :)."