from .service_list import ServiceList
from .apologies import Apologies
from .action import Action


class OOV(Action):
    def __init__(self):
        self.apologies = Apologies()
        self.service_list = ServiceList()

    def concatenate_strings(self, strings):
        return ' '.join(strings)

    def run(self, action_obj=None):
        return self.concatenate_strings(
            [self.apologies.run(), "I don't understand your request. "]) + self.service_list.run()
