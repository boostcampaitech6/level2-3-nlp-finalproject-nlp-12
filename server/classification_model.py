import os
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn

from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import ElectraModel, ElectraConfig, ElectraForPreTraining, PreTrainedModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from datasets import Dataset

from dotenv import load_dotenv
import ast
from loguru import logger
import time

load_dotenv()

def load_classifier():
    model_config = ast.literal_eval(os.getenv("CLASSIFIER_MODEL_CONFIG"))

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_path'])
    tokenizer.add_special_tokens({"additional_special_tokens" : ["[Q1]", "[Q2]","[Q3]", "[Q4]","[Q5]"]})
    
    # model setting
    config = ElectraConfig.from_pretrained(model_config['model_path'])
    model = CustomModel.from_pretrained(
        model_config['model_path'],
        num_labels=2,
        addi_feat_size=model_config['addi_feat_size'],
        config = config,
        tokenizer = tokenizer
        )
    model.eval()
    return model, tokenizer

def load_classifiers():
    model_config = ast.literal_eval(os.getenv("CLASSIFIER_MODELS_CONFIG"))

    models = []
    for model_path in model_config['model_path']:
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model setting
        config = ElectraConfig.from_pretrained(model_path)
        model = CustomModel.from_pretrained(
            model_path,
            num_labels=2,
            addi_feat_size=model_config['addi_feat_size'],
            config = config,
            tokenizer = tokenizer
            )
        model.eval()
        models.append(model)

    logger.info("load_classifiers complete")
    return models, tokenizer
    

def preprocessing_special_token(tokenizer, text, age, gender, q_num, max_len=512):
    q_prom = "[Q" + str(q_num) + "] "
    gender_str = "남자" if gender==1 else "여자"
    prompt = "화자의 나이는 {}세고, 성별은 {}입니다. 다음은 화자의 상담 내용입니다 : ".format(age, gender_str)
    final_text = q_prom + prompt + text
    return tokenizer(final_text, padding='max_length', truncation=True, return_tensors="pt", max_length=max_len, add_special_tokens=True)


def preprocessing_prompt(tokenizer, text, age, gender, q_num, max_len=512):
    question = ["행복한 기억", "불행한 기억", "긍정적 사진", "중립적 사진", "부정적 사진"]
    gender_str = "남자" if gender==1 else "여자"
    prompt = "화자의 나이는 {}세고, 성별은 {}입니다. 다음은 {}에 대한 화자의 이야기입니다 : ".format(age, gender_str, question[q_num-1])
    final_text = prompt + text
    return tokenizer(final_text, padding='max_length', truncation=True, return_tensors="pt", max_length=max_len, add_special_tokens=True)

class TextDataset(Dataset):
    def __init__(self, predict_df, processing_type, tokenizer):
        self.tokenizer = tokenizer
        self.dataset = predict_df
        self.addi_feat_df = self.dataset.drop(columns=["age", "gender", "stt", "question"])
        self.inputs = []

        for idx in range(len(self.dataset)):

            try :
                if processing_type == "special_token":
                    tokenized = preprocessing_special_token(self.tokenizer, self.dataset["stt"][idx], self.dataset["age"][idx], self.dataset["gender"][idx], self.dataset["question"][idx])
                elif processing_type == "prompt":
                    tokenized = preprocessing_prompt(self.tokenizer, self.dataset["stt"][idx], self.dataset["age"][idx], self.dataset["gender"][idx], self.dataset["question"][idx])
                
                self.inputs.append(tokenized)
            except :
                print(f"Error occured at {idx}th data")
                pass

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'addi_feat' : self.addi_feat_df.loc[idx].values,
        }


class CustomModel(PreTrainedModel) :
    def __init__(self, config, num_labels, addi_feat_size, tokenizer) :
        super().__init__(config=config)
        self.backbone = ElectraForPreTraining(config=config) # ??
        # self.backbone.resize_token_embeddings(len(tokenizer))
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear((768 + addi_feat_size), self.num_labels)        

    def forward(self, input_ids, attention_mask, addi_feat, labels=None) :
        backbone_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
        x = self.dropout(backbone_output["hidden_states"][-1][:,0,:])
        x = self.concat_features(x, addi_feat)
        x = self.classifier(x)
        logits = torch.nn.functional.softmax(x, dim=1)

        if labels is not None :
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.Tensor(labels).long().view(-1))
        else :
            loss = 0

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_output.hidden_states,
            attentions=backbone_output.attentions,
        )
    
    def concat_features(self, x, addi_feat) :
        addi_feat = torch.log(torch.abs(torch.Tensor(addi_feat+1)))
        return torch.cat((x, addi_feat), dim=1)


def predict_classification(predict_df, model, tokenizer):
    start_time = time.time()

    processing_type = os.getenv("PROCESSING_TYPE")

    dataset = TextDataset(predict_df, processing_type, tokenizer)

    input_ids = torch.stack([dataset[0]["input_ids"]], dim=0)
    attention_mask = torch.stack([dataset[0]["attention_mask"]], dim=0)
    addi_feat = torch.stack([torch.Tensor(dataset[0]["addi_feat"])])

    logits = model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        addi_feat = addi_feat,
    ).logits

    logger.info(f"Classification time taken: {time.time() - start_time}")

    return logits[0][1].item()