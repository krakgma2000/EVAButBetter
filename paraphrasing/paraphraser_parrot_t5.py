from paraphrasing.paraphraser import Paraphraser
from parrot import Parrot
# import torch
import warnings

warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

MODEL_NAME = "igorktech/ent5-base-paraphraser"  # "ceshine/t5-paraphrase-paws-msrp-opinosis"#"prithivida/parrot_paraphraser_on_T5"


class ParaphraserParrot(Paraphraser):
    def __init__(self, model=Parrot(model_tag=MODEL_NAME)):
        super().__init__(model)

    def paraphrase(self, sentences, **kwargs):
        if isinstance(sentences, list):
            return [self.model.augment(input_phrase=sentence, **kwargs) for sentence in sentences]
        return self.model.augment(input_phrase=sentences, **kwargs)

# phrases = "Can you recommend some upscale restaurants in Newyork?"#["Can you recommend some upscale restaurants in Newyork?",
# #            "Can you recommend some upscale restaurants in Newyork?"
# #            ]
# import time
# tm = time.time()
# print(ParaphraserParrot().paraphrase(phrases, max_return_phrases=5, use_gpu=False))
# print(time.time()-tm)
