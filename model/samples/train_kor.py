import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer

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
    parser.add_argument('--data_dir', type=str, default="./data/kor_sample.csv")
    parser.add_argument('--output_dir', type=str, default="./output_kor")
    parser.add_argument('--dev_ratio', type=float, default=0.2)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    return args


### 데이터셋 구성
class SampleDataset(Dataset):

    def __init__(self, data_dir, tokenizer):
        
        self.data = pd.read_csv(data_dir)
        self.labels = self.data['target'].tolist()
        self.inputs = []

        for text in tqdm(self.data["text"]) :
            tmp = tokenizer(
                text,
                max_length=512,
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
            # 'token_type_ids': self.inputs[idx]['token_type_ids'].squeeze(),
            'labels': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)


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

    model_name = "kykim/bert-kor-base"
    # kykim/albert-kor-base
    # monologg/DistilKoBert

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id).to(DEVICE)
    

    # 데이터셋 생성
    dataset = SampleDataset(args.data_dir, tokenizer)
    dataset_train, dataset_dev = train_test_split(dataset, test_size=args.dev_ratio, random_state=SEED)

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
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained('./best_model')