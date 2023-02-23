from .action import Action


class Spelling(Action):
    def run(self,action_obj):
        word = action_obj["word"]
        output = "The word " + word + " is spelled as: "
        for character in word:
            output += character + " - "
        return output[:-3]