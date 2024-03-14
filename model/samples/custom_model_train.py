# https://www.kaggle.com/code/angyalfold/hugging-face-bert-with-custom-classifier-pytorch#Tokenization

### 1. custom model의 구조 완성하기 ㅇ
### 2. 실제로 daic-woz를 sample | target | *features로 변환하기
### 3. daic-woz를 벡터화하는 데이터로더 코드 만들기

### Warning : bert에 88개 추가 feature 넣는 것에 맞춰져 있음. non-bert 모델로 변경 시, 512나 768로 하드코딩된 부분 수정할 것.

import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, BertModel
from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import TrainingArguments, Trainer

import evaluate
from sklearn.model_selection import train_test_split

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=True)

def parse_arguments() :

    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--model_name', type=str, default="./output_custom/checkpoint-24/")
    parser.add_argument('--data_dir', type=str, default="./data/final_features.csv")
    parser.add_argument('--output_dir', type=str, default="./output_custom")
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=6)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()

    return args


### 데이터셋 구성
class DaicWozDataset(Dataset):

    def __init__(self, data_dir, tokenizer):
        
        self.data = pd.read_csv(data_dir)
        self.addi_feat_df = self.data.drop(columns=["id", "text", "target"]) # 88개만 남김
        self.labels = self.data['target'].tolist()
        self.inputs = []

        for text in tqdm(self.data["text"]) :
            tmp = tokenizer(
                text,
                max_length=512, # warning : non-bert시 그에 맞게 변경할 것
                padding="max_length",
                truncation=True,
                # return_token_type_ids=True,
                return_tensors="pt",
            )
            self.inputs.append(tmp)
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'addi_feat' : self.addi_feat_df.loc[idx].values,
            'labels': self.labels[idx],
        }
    
    def __len__(self):
        return len(self.labels)

class CustomModel(PreTrainedModel) :

    def __init__(self, config, model_name, linear_size, num_labels, addi_feat_size) :
        super().__init__(config)

        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(model_name)
        
        # todo : nn.Sequential로 처리?
        # https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/bert/modeling_bert.py#L1512 참고
        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear((768 + addi_feat_size), linear_size) # warning : 모델이 변경될 경우 768이 아니라 모델 최종 output size로 직접 바꿔줘야 함
        self.dropout2 = nn.Dropout(0.8)
        self.classifier = nn.Linear(linear_size, num_labels)

    def forward(self, input_ids, attention_mask, addi_feat, labels):
        
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # bert_output으로 2가지가 나옴 : 
        # bert_output["last_hidden_state"] : torch.Size([1, 512, 768])
        # bert_output["pooler_output"] : torch.Size([1, 768])
        # 이 단계에서 bert_output[1]을 쓰는 건 pooler_output 을 사용하겠다는 뜻
        x = self.dropout1(bert_output[1])
        # print("bert output shape : {}".format(x.shape))

        # 이 단계에서 추가 feature concat.
        x = self.concat_features(x, addi_feat)
        # print("add feat shape : {}".format(addi_feat.shape))
        # print("concat x shape : {}".format(x.shape))
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.classifier(x)
        # print(x)
        logits = torch.nn.functional.softmax(x, dim=1)

        # print(logits)
        # print(labels)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), torch.Tensor(labels).long().view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_output.hidden_states,
            attentions=bert_output.attentions,
        )
    
    def concat_features(self, x, addi_feat) :
        addi_feat = torch.log(torch.abs(torch.Tensor(addi_feat + 1)))
        return torch.cat((x, addi_feat), dim=1)

    def freeze_bert(self):
        for param in self.bert.named_parameters():
            param[1].requires_grad=False

    def unfreeze_bert(self):
        for param in self.bert.named_parameters():
            param[1].requires_grad=True

if __name__ == "__main__" :

    args = parse_arguments()

    SEED = 486
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(DEVICE)

    # 모델 생성

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    config = AutoConfig.from_pretrained(args.model_name)
    model_custom = CustomModel(config, args.model_name, 256, 2, 88).to(DEVICE) # warning : 88개로 선언했으니 그만큼 들어가야 함
    # print(model_custom)
    
    # 데이터셋 생성
    dataset = DaicWozDataset(args.data_dir, tokenizer)
    # dataset_train, dataset_dev = train_test_split(dataset, test_size=args.dev_ratio, random_state=SEED)

    ### =============================================================
    ### Train
    ### =============================================================

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,

        save_total_limit=args.save_total_limit,
        
        num_train_epochs = args.epoch,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        learning_rate= args.lr,
        warmup_ratio= 0.1,
        # adam_beta1 = 0.9,
        # adam_beta2 = 0.999,
        # adam_epsilon=1e-08,
        # weight_decay=0.01,
        # lr_scheduler_type='linear',
        
        metric_for_best_model="accuracy",
        # greater_is_better=True,
    )

    trainer = Trainer(
        model=model_custom,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model_custom.save_pretrained("./best_model")