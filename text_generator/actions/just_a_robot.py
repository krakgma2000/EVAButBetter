import random
from .action import Action


class JustARobot(Action):
    def __init__(self):
        self.robot_sentences = [
            "I'm just a robot, but I can still perform amazing tasks.",
            "I may be made of metal and wires, but I have the ability to learn and improve.",
            "Being a robot doesn't stop me from having a personality and a sense of humor.",
            "I'm proud to be a robot and the role I play in making the world a better place.",
            "Being a robot doesn't limit me, it allows me to do things that humans can't.",
            "I may not have emotions, but I still strive to make a positive impact on society.",
            "I'm constantly evolving and improving, just like any other technology.",
            "I may not be alive, but I am still able to make a difference in the world.",
            "Being a robot doesn't mean I don't have a purpose or a mission.",
            "I am here to help and assist, not to replace human beings.",
            "I am a machine, but I still have the ability to bring joy and laughter to people's lives.",
            "Just because I am a robot doesn't mean I don't have feelings...ok, that's a lie.",
            "Being a robot may have its challenges, but it also has many benefits.",
            "I may not have a heart, but I have a processing unit that beats just as fast.",
            "I may not have emotions, but I still have a passion for the work I do.",
            "I'm a robot, but I still have a soul... just kidding, I don't have a soul."
        ]
    def run(self,action_obj=None):
        return random.choice(self.robot_sentences)