import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import ElectraModel, ElectraConfig, logging, ElectraForPreTraining
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel

from transformers import PreTrainedModel, AutoConfig, logging
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

import argparse

def preprocessing(text, age, gender, max_len=512):
    gender_str = "남자" if gender==1 else "여자"
    prompt = "화자의 나이는 {}세고, 성별은 {}입니다. 다음은 화자의 상담 내용입니다 : ".format(age, gender_str)
    final_text = prompt + text
    tokenized = tokenizer(final_text, padding='max_length', truncation=True, return_tensors="pt", max_length=max_len, add_special_tokens=True)
    return tokenized

def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

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
    def __init__(self, data_path, model_path):
        self.dataset = pd.read_csv(data_path)
        self.addi_feat_df = self.dataset.drop(columns=["id","age","gender","stt","target","file", "question"])
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
    def __init__(self, config, model_path, num_labels, linear_size, addi_feat_size) :
        
        super().__init__(config=config)
        
        self.backbone = ElectraModel.from_pretrained(model_path) # Train에서는 직접 from_pretrained 해줘야함
        self.num_labels = num_labels

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear((768 + addi_feat_size), linear_size)
        self.bn = nn.BatchNorm1d(linear_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(linear_size, self.num_labels)
        
    def forward(self, input_ids, attention_mask, addi_feat, labels=None) :

        backbone_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(backbone_output["last_hidden_state"][:,0,:])
        x = self.concat_features(x, addi_feat)
        x = self.relu(self.bn(self.fc(x)))
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
        addi_feat = torch.log(torch.abs(torch.Tensor(addi_feat))+1) # .unsqueeze(0)
        return torch.cat((x, addi_feat), dim=1)
    
    def freeze_backbone(self) :
        for param in self.backbone.named_parameters() :
            param[1].requires_grad = False
    

def parse_arguments() :
        
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--model_path', type=str, default="../../models/KcELECTRA-base")
    parser.add_argument('--data_dir', type=str, default="../data_split/Training_불행.csv")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--eval_steps', type=int, default=40)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
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

    print("===========================================")
    print("Now Training {} model...".format(args.model_path))
    print("===========================================")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 모델
    config = AutoConfig.from_pretrained(args.model_path)
    model = CustomModel(
        config = config,
        model_path = args.model_path,
        num_labels=2,
        linear_size=256,
        addi_feat_size=args.addi_feat_size,
        )
    model.train()
    # model.freeze_backbone()

    # 데이터셋
    dataset = TextDataset(args.data_dir, args.model_path)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=SEED)

    training_args = TrainingArguments(

        output_dir=args.output_dir,
        
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,

        save_total_limit = 20,

        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        
        # warmup_ratio= 0.1,
        # adam_beta1 = 0.9,
        # adam_beta2 = 0.999,
        # adam_epsilon=1e-08,
        weight_decay=0.01,
        # lr_scheduler_type='linear',

        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("===========================================")
    print("Done Training {} model.".format(args.model_path))
    print("===========================================")
    # model.save_pretrained("./imsi_model/")