from .service_list import ServiceList
from .apologies import Apologies
from .action import Action


class OOV(Action):
    def __init__(self, t5, tokenizer):
        self.apologies = Apologies()
        self.service_list = ServiceList()
        self.t5 = t5
        self.tokenizer = tokenizer

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

    def run(self, action_obj=None):
        return self.concatenate_strings(
            [self.apologies.run(), "I don't understand your request. "]) + self.service_list.run()
