
from datasets import load_dataset
from transformers import AdamW
from transformers import Wav2Vec2ForSequenceClassification
from transformers import EvalPrediction

#Pretrained 된 wav2vec2 가져오기
class EmotionClassifier(Wav2Vec2ForSequenceClassification):
    def __init__(self):
        super().__init__(
            config=Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h"),
            num_labels=7,
        )

# 모델 생성
model = EmotionClassifier()


# 데이터셋 로드
dataset = load_dataset("")

# 데이터 전처리
def preprocess_data(data):
    audio = data["audio"]
    ""

# 콜레이터
def collate_fn(batch):
    # 
    return {
        "input_values": batch["input_values"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

# 데이터 로더
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)


# optimizer 및 loss 함수 정의
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# 모델 학습
for epoch in range(100000):
    for batch in train_loader:
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# 모델 평가
def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in val_loader:
            outputs = model(**batch)
            predictions.append(EvalPrediction(logits=outputs.logits))

# 평가 코드 실행
evaluate(model, val_loader)