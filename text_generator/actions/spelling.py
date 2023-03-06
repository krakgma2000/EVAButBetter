from .action import Action


class Spelling(Action):
    def run(self,action_obj):
        word = action_obj["word"]
        if word is None:
            return "Which word would you like to spell?"
        else:
            output = "The word " + word + " is spelled as: "
            for character in word:
                output += character + " - "
            return output[:-3]