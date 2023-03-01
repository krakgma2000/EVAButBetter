from paraphrasing.paraphraser import Paraphraser
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import warnings
import torch

warnings.filterwarnings("ignore")

# "ceshine/t5-paraphrase-paws-msrp-opinosis" #"prithivida/parrot_paraphraser_on_T5"
MODEL_NAME = "igorktech/ent5-base-paraphraser"
TASK_PREFIX = 'paraphrase | '


class T5Model():
    def __init__(self, model_name="igorktech/ent5-base-paraphraser"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)


class ParaphrasetT5(Paraphraser):
    def __init__(self, model=T5Model(MODEL_NAME), task_prefix=TASK_PREFIX):
        super().__init__(model)
        self.task_prefix = task_prefix

    def paraphrase(self, sentences, gpu=False, **kwargs):
        if gpu:
            device = "cuda:0"
        else:
            device = "cpu"
        self.model.model = self.model.model.to(device)
        self.model.model.eval()
        if isinstance(sentences, list):
            response_list = []
            sentences = [self.task_prefix + phrase for phrase in sentences]
            for sentence in sentences:
                inputs = self.model.tokenizer(sentence, return_tensors='pt').to(device)
                with torch.no_grad():
                    hypotheses = self.model.model.generate(**inputs, **kwargs)
                response_list.append(self.model.tokenizer.batch_decode(hypotheses, skip_special_tokens=True))
            return response_list

        sentences = self.task_prefix + sentences
        inputs = self.model.tokenizer(sentences, return_tensors='pt').to(device)
        with torch.no_grad():
            hypotheses = self.model.model.generate(**inputs, **kwargs)
        return self.model.tokenizer.batch_decode(hypotheses, skip_special_tokens=True)

# import time
# tm = time.time()
# print('model: ',
#       ParaphrasetT5().paraphrase('Can you recommend some upscale restaurants in Newyork?', do_sample=True,
#                                  top_p=0.95,
#                                  num_return_sequences=4,
#                                  # repetition_penalty=2.0,
#                                  # num_beams = 4,
#                                  # num_beam_groups = 4,
#                                  # diversity_penalty = 2.0,
#                                  top_k=50,
#                                  early_stopping=True,
#                                  max_length=64))
# print(time.time() - tm)
