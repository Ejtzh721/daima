import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 屏蔽未来类警告
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示问题
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义和训练时一样的模型结构
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

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# 读取测试集
test_df = pd.read_csv("mnist_test.csv")

# 随机选一张图片来预测
idx = 400#输入一个想要测试的图片
label = test_df.iloc[idx, 0]
pixels = test_df.iloc[idx, 1:].values / 255.0
img = pixels.reshape(1, 1, 28, 28)  # 转为模型输入格式

# 预测
with torch.no_grad():
    output = model(torch.tensor(img, dtype=torch.float32).to(device))
    pred = torch.argmax(output, dim=1).item()

# 显示图片和结果
plt.imshow(pixels.reshape(28, 28), cmap="gray")
plt.title(f"真实标签: {label}, 预测结果: {pred}")
plt.axis("off")
plt.show()
print(f"第 {idx} 张：真实标签是 {label}，模型预测结果是 {pred}")