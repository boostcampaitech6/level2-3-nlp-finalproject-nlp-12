import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn

from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import ElectraModel, ElectraConfig, logging, ElectraForPreTraining
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers import PreTrainedModel, AutoConfig, ElectraConfig

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

import argparse

def preprocessing(text, age, gender, max_len=512):
    gender_str = "남자" if gender==1 else "여자"
    prompt = "화자의 나이는 {}세고, 성별은 {}입니다. 다음은 화자의 상담 내용입니다 : ".format(age, gender_str)
    final_text = prompt + text
    return tokenizer(final_text, padding='max_length', truncation=True, return_tensors="pt", max_length=max_len, add_special_tokens=True)

# def preprocessing(text, max_len=512):
#     return tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=max_len, add_special_tokens=True)

def compute_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    return {
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "f1_score" : f1,
        }

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
        self.addi_feat_df = self.dataset.drop(columns=["id","age","gender","stt","target","file","question"])
        self.inputs = []
        self.labels = []

        for idx in tqdm(range(len(self.dataset))):

            try :
                tokenized = preprocessing(self.dataset["stt"][idx], self.dataset["age"][idx], self.dataset["gender"][idx])
                self.inputs.append(tokenized)
                self.labels.append(self.dataset['target'][idx])
            except :
                pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'addi_feat' : self.addi_feat_df.loc[idx].values,
            'labels': self.labels[idx],
        }


class CustomModel(PreTrainedModel) :
    def __init__(self, config, num_labels, addi_feat_size) :
        super().__init__(config=config)
        self.backbone = ElectraForPreTraining(config=config) # infer에서는 아키텍쳐만 잡아주면 됨
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
        # print(logits)

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
        addi_feat = torch.log(torch.abs(torch.Tensor(addi_feat+1))) # .unsqueeze(0)
        return torch.cat((x, addi_feat), dim=1)
    
def parse_arguments() :
        
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--model_path', type=str, default="./output_그림3/checkpoint-400/")
    parser.add_argument('--data_dir', type=str, default="../data_split/Validation_그림3.csv")
    parser.add_argument('--addi_feat_size', type=int, default=88)
    args = parser.parse_args()

    return args


if __name__ == "__main__" :


    args = parse_arguments()

    SEED = 486
    random.seed(SEED)
    np.random.seed(SEED)
    torch.autograd.set_detect_anomaly(True)

    logging.set_verbosity_info()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # tokenizer.add_special_tokens({"additional_special_tokens" : ["[Q1]", "[Q2]","[Q3]", "[Q4]","[Q5]"]})
    
    # model setting
    config = ElectraConfig.from_pretrained(args.model_path)
    model = CustomModel.from_pretrained(
        args.model_path,
        num_labels=2,
        addi_feat_size=args.addi_feat_size,
        config = config,
        )
    model.eval()

    # dataset 

    dataset = TextDataset(args.data_dir)

    input_ids_all = torch.stack([dataset[i]["input_ids"] for i in range(len(dataset))], dim=0)
    attention_mask_all = torch.stack([dataset[i]["attention_mask"] for i in range(len(dataset))], dim=0)
    addi_feat_all = torch.stack([torch.Tensor(dataset[i]["addi_feat"]) for i in range(len(dataset))])
    labels = [dataset[i]["labels"] for i in range(len(dataset))]

    preds = []
    batch_size = 16

    for i in tqdm(range(0, len(dataset), batch_size)) :
        
        input_ids_batch = input_ids_all[i:i+batch_size, :]
        attention_mask_batch = attention_mask_all[i:i+batch_size, :]
        addi_feat_batch = addi_feat_all[i:i+batch_size, :]

        logits = model(
            input_ids = input_ids_batch,
            attention_mask = attention_mask_batch,
            addi_feat = addi_feat_batch,
        ).logits

        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(pred)

    print(compute_metrics(preds, labels))