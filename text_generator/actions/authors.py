from .action import Action


class Authors(Action):
    def run(self, action_obj=None):
        return "I was created by three UPF students: Igor, Guillem and Dongjun."