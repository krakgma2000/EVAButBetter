from .greeting import Greeting
from .service_list import ServiceList
from .action import Action


class Introduction(Action):
    def __init__(self):
        self.greeting = Greeting()
        self.service_list = ServiceList()
    def run(self,action_obj=None):
        return "Let me introduce myself. My name is EVA, and I'm the official UPF virtual assistant. " + self.service_list.run()
