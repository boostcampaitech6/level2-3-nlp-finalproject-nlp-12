import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(88, 44),
            nn.Tanh(),
            nn.Linear(44, 22),
            nn.Tanh(),
            nn.Linear(22, 11),
            nn.Tanh(),
            nn.Linear(11, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 11),
            nn.Tanh(),
            nn.Linear(11, 22),
            nn.Tanh(),
            nn.Linear(22, 44),
            nn.Tanh(),
            nn.Linear(44, 88),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.data = X.values
        self.target = y['target'].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        X = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.target[index], dtype=torch.float32)
        return X, y

# 데이터 준비
data_dir = "path/to/data.csv"
df = pd.read_csv(data_dir)
X_df = df.iloc[:, 6:]
y_df = df.iloc[:, :6]
y_df = y_df.reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.1, random_state=42)



# 데이터셋 데이터로더
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)



# 모델 학습
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 10000
best_loss = 1000
for epoch in range(num_epochs):
    for (features, _), (test_features, _) in zip(train_loader, test_loader):
        model.train()
        _, output = model(features)
        loss = criterion(output, features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        eval_loss = criterion(model(test_features)[1], test_features)
    
    print(f'Epoch: [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Eval Loss: {eval_loss.item():.4f}')
    if eval_loss < best_loss:
        torch.save(model, 'best-model.pt')
        torch.save(model.state_dict(), 'best-model-parameters.pt')
        best_loss = eval_loss

print("학습완료")



# latent vector 뽑기
latent_vectors = []
labels = []

with torch.no_grad():
    for features, lbls in test_loader:
        latent, _ = model(features)
        latent_vectors.append(latent)
        labels.append(lbls)

latent_vectors = torch.cat(latent_vectors, dim=0)
labels = torch.cat(labels, dim=0)

print(latent_vectors)
print(labels)