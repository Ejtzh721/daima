# 导入所需库
# pandas：读取CSV格式的MNIST数据
# numpy：数值计算
# torch：PyTorch深度学习框架
# nn：构建神经网络的核心模块
# Dataset/DataLoader：用于封装和批量加载数据
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- 1. 超参数设置 ----------------------
# 每批次训练的样本数量
batch_size = 64
# 学习率（控制模型参数更新步长）
lr = 0.001
# 训练轮数（整个数据集遍历几次）
epochs = 5
# 指定训练设备：有GPU用GPU，没有用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 2. 自定义数据集类 ----------------------
# 继承PyTorch的Dataset，必须实现 __init__, __len__, __getitem__
class MNISTDataset(Dataset):
    # 初始化方法：读取数据、划分特征和标签
    def __init__(self, csv_path, train=True):
        # 读取CSV文件
        self.df = pd.read_csv(csv_path)
        self.train = train
        # 训练集和测试集处理逻辑一样：第一列是标签，后面是像素
        if train:
            # 标签：第0列
            self.labels = self.df.iloc[:, 0].values
            # 图像：第1列之后所有像素，转为浮点型并归一化到0~1
            self.images = self.df.iloc[:, 1:].values.astype(np.float32) / 255.0
        else:
            self.labels = self.df.iloc[:, 0].values
            self.images = self.df.iloc[:, 1:].values.astype(np.float32) / 255.0

    # 返回数据集总样本数
    def __len__(self):
        return len(self.images)

    # 根据索引idx获取一条数据
    def __getitem__(self, idx):
        # 将一维像素(784) reshape 成 CNN需要的格式：(通道, 高, 宽) = (1,28,28)
        img = self.images[idx].reshape(1, 28, 28)
        label = self.labels[idx]
        # 转为PyTorch张量，标签类型为long
        return torch.tensor(img), torch.tensor(label, dtype=torch.long)

# ---------------------- 3. 加载数据 ----------------------
# 加载训练集
train_dataset = MNISTDataset("mnist_train.csv", train=True)
# 加载测试集
test_dataset = MNISTDataset("mnist_test.csv", train=False)

# 训练集数据加载器：打乱顺序、分批返回
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 测试集数据加载器：不打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------- 4. 定义简单 CNN 模型 ----------------------
# 继承nn.Module，自定义卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积：输入通道1(灰度图)，输出通道32，卷积核3x3，padding=1保持尺寸不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二层卷积：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化：2x2池化窗口，步长2（尺寸减半）
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：输入 64*7*7（两次池化后），输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：输出10分类（0~9数字）
        self.fc2 = nn.Linear(128, 10)
        # 激活函数ReLU
        self.relu = nn.ReLU()

    # 前向传播：数据流经网络的路径
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        # 卷积 -> 激活 -> 池化
        x = self.pool(self.relu(self.conv2(x)))
        # 展平：将特征图转为一维向量，用于全连接层
        x = x.view(-1, 64 * 7 * 7)
        # 全连接+激活
        x = self.relu(self.fc1(x))
        # 最后一层输出10分类得分（不用softmax，CrossEntropyLoss会自动处理）
        x = self.fc2(x)
        return x

# 实例化模型并移动到指定设备（GPU/CPU）
model = SimpleCNN().to(device)
# 损失函数：交叉熵损失（分类任务标配）
criterion = nn.CrossEntropyLoss()
# 优化器：Adam自适应优化器，传入模型参数和学习率
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ---------------------- 5. 训练循环 ----------------------
print("开始训练...")
# 开启训练模式（启用BatchNorm、Dropout等）
model.train()
# 遍历每一轮
for epoch in range(epochs):
    total_loss = 0.0
    # 遍历训练集的每一个批次
    for imgs, labels in train_loader:
        # 数据移到设备上
        imgs, labels = imgs.to(device), labels.to(device)

        # 前向传播：模型输出预测结果
        outputs = model(imgs)
        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播与参数更新三步曲
        optimizer.zero_grad()  # 清空上一轮梯度
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 根据梯度更新参数

        # 累计损失
        total_loss += loss.item()

    # 计算本轮平均损失
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# ---------------------- 6. 测试模型准确率 ----------------------
# 开启评估模式（关闭Dropout、BatchNorm等）
model.eval()
correct = 0   # 预测正确的样本数
total = 0     # 总样本数
# 测试时不计算梯度，节省显存/加速
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        # 取输出最大值的下标作为预测类别
        _, predicted = torch.max(outputs.data, 1)
        # 累计总样本数
        total += labels.size(0)
        # 累计正确预测数
        correct += (predicted == labels).sum().item()

# 计算并打印准确率
print(f"测试集准确率: {100 * correct / total:.2f}%")

# ---------------------- 7. 保存模型 ----------------------
# 只保存模型参数（推荐方式，体积小、易加载）
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已保存为 mnist_cnn.pth")
