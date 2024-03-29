from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import pandas as pd
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split

def preprocessing(text):
    return tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=512)

def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class TextDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
        self.inputs = []
        self.labels = []

        for idx, text in enumerate(self.dataset['text']):
            tokenized = preprocessing(text)
            self.inputs.append(tokenized)
            self.labels.append(self.dataset['target'][idx])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'labels': self.labels[idx]
        }
    

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

dataset = TextDataset('../data/text_with_features.csv')
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load('accuracy')

train_dataset, test_dataset = train_test_split(dataset, test_size=0.1)

training_args = TrainingArguments(
    output_dir="../output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
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

# model.save_pretrained("../output")