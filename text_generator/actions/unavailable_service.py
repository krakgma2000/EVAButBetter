from .introduction import Introduction
from .apologies import Apologies
from .action import Action


class UnavailableService(Action):
    def __init__(self,tokenizer,t5):
        self.apologies = Apologies()
        self.introduction = Introduction()
        self.tokenizer = tokenizer
        self.t5 = t5

    def concatenate_strings(self, strings):
        # Load T5 tokenizer and model
        # tokenizer = T5Tokenizer.from_pretrained('t5-base')
        # model = T5ForConditionalGeneration.from_pretrained('t5-base')

        # Tokenize the strings and join them with separator tokens
        inputs = self.tokenizer.encode(' ; '.join(strings), return_tensors='pt')

        # Generate the concatenated string
        output = self.t5.generate(inputs, max_length=512)

        # Decode the generated output
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded_output

    def run(self, action_obj):
        service = action_obj["service"]
        run_intro = action_obj["run_intro"]
        apologies = self.apologies.run()
        standard_sentence = "I can't help you with your request."
        if service == "location":
            to_return = self.concatenate_strings([apologies, "I can't access to your current location",standard_sentence])
        elif service == "no_internet":
            to_return = self.concatenate_strings([apologies, "My internet connection is very limited",standard_sentence])
        else:
            to_return = self.concatenate_strings([apologies, "I can't handle " + service + " services or information",standard_sentence])

        if run_intro:
            to_return += " "
            to_return += self.introduction.run()

        return to_return
