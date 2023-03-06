from PyDictionary import PyDictionary
from .action import Action


class Definition(Action):
    def __init__(self):
        self.dc = PyDictionary()

    def run(self, action_obj):
        word = action_obj["word"]
        if word is None:
            return "Which word do you want to know the meaning for?"
        else:
            mn = self.dc.meaning(word)
            mn = list(mn.values())[0][0]
            return "This is what I found about the word \"" + word + "\": " + mn
