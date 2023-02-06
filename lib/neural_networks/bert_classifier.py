from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
LABELS = ['fh', 's', 'b', '%', 'qy', 'fg', 'qw', 'qrr', 'h', 'qr', 'qo', 'qh']
model_bert = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                          os.path.join("models", 'mrda_bert_312_10ep.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BERTModel(torch.nn.Module):
    def __init__(self, model, output_size):
        super(BERTModel, self).__init__()
        self.bert = model  # transformers.BertModel.from_pretrained(model_name)#huawei-noah/TinyBERT_General_6L_768D
        self.dropout1 = torch.nn.Dropout(0.3)
        self.l2 = torch.nn.Linear(self.bert.config.hidden_size, output_size)
        self.relu1 = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)

        output = self.dropout1(output)
        output = self.l2(output)
        output = self.relu1(output)
        return output


# MODEL_NAME = 'huawei-noah/TinyBERT_General_6L_768D'#'huawei-noah/TinyBERT_General_4L_312D'#huawei-noah/TinyBERT_General_6L_768D

model = BERTModel(model=model_bert, output_size=len(LABELS))


class BERTClassifier:
    def __init__(self, model=model, tokenizer=tokenizer, model_path=PATH, labels=LABELS):
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.tokenizer = tokenizer
        self.labels = labels

    def predict(self, text, return_type='label'):
        """ Calculate dialog act type of a text. `return_type` can be 'label' or 'proba' """
        with torch.no_grad():
            inputs = self.tokenizer(text, padding='max_length',
                                    max_length=256, truncation=True, add_special_tokens=True,
                                    pad_to_max_length=True, return_tensors="pt", )
            ids = inputs.input_ids
            ids = ids.to(device)
            mask = inputs.attention_mask
            mask = mask.to(device)
            # token_type_ids = inputs.token_type_ids
            # token_type_ids = token_type_ids.to(device)
            outputs = self.model(ids, mask)
            proba = F.softmax(outputs).cpu().numpy()[0]

        if return_type == 'label':
            return self.labels[proba.argmax()]
        elif return_type == 'index':
            return proba.argmax()
        elif return_type == 'proba':
            max(proba)
        return max(proba)


# import time
#
# bert = BERTClassifier()
# time_start = time.time()
# print(bert.predict("Where are my pants?"))
# print(time.time()-time_start)