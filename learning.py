import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- 1. 超参数设置 ----------------------
batch_size = 64
lr = 0.001
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 2. 自定义数据集类 ----------------------
class MNISTDataset(Dataset):
    def __init__(self, csv_path, train=True):
        self.df = pd.read_csv(csv_path)
        self.train = train
        if train:
            self.labels = self.df.iloc[:, 0].values
            self.images = self.df.iloc[:, 1:].values.astype(np.float32) / 255.0
        else:
            self.labels = self.df.iloc[:, 0].values
            self.images = self.df.iloc[:, 1:].values.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(1, 28, 28)  # 转为 (1, 28, 28) 的灰度图
        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(label, dtype=torch.long)

# ---------------------- 3. 加载数据 ----------------------
train_dataset = MNISTDataset("mnist_train.csv", train=True)
test_dataset = MNISTDataset("mnist_test.csv", train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------- 4. 定义简单 CNN 模型 ----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------------- 5. 训练循环 ----------------------
print("开始训练...")
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向传播 + 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# ---------------------- 6. 测试模型准确率 ----------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")

# ---------------------- 7. 保存模型 ----------------------
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已保存为 mnist_cnn.pth")