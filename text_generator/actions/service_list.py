from .action import Action


class ServiceList(Action):
    def run(self,action_obj=None):
        return "I can help you with any question you have about the university, for example:\n" \
               "* In which room can I find professor xxxx?\n" \
               "* How many faculties does the upf have?\n" \
               "* Does the UPF have a Computer Science course?\n" \
               "* Which masters does the UPF offer?\n" \
               "* At which time does the library close?"
