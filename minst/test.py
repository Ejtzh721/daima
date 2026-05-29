# 导入警告模块，屏蔽指定类型警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 屏蔽未来类警告
# 导入绘图库
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示问题
# 导入数据处理与深度学习相关库
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义和训练时一样的模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 第一层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第二层卷积
        self.pool = nn.MaxPool2d(2, 2)                             # 最大池化层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                      # 全连接层1
        self.fc2 = nn.Linear(128, 10)                              # 全连接层2，输出10分类
        self.relu = nn.ReLU()                                      # 激活函数

    # 前向传播
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 特征图展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定运行设备
model = SimpleCNN().to(device)                                         # 初始化模型并迁移至对应设备
model.load_state_dict(torch.load("mnist_cnn.pth"))                     # 载入训练好的模型参数
model.eval()                                                           # 切换为评估模式

# 读取测试集
test_df = pd.read_csv("mnist_test.csv")  # 读取测试集CSV文件

# 随机选一张图片来预测
idx = 400                                # 指定待测试图片索引
label = test_df.iloc[idx, 0]             # 获取图片真实标签
pixels = test_df.iloc[idx, 1:].values / 255.0  # 提取像素数据并归一化
img = pixels.reshape(1, 1, 28, 28)      # 转为模型输入格式

# 预测
with torch.no_grad():                    # 关闭梯度计算
    output = model(torch.tensor(img, dtype=torch.float32).to(device))  # 模型前向推理
    pred = torch.argmax(output, dim=1).item()  # 获取预测类别

# 显示图片和结果
plt.imshow(pixels.reshape(28, 28), cmap="gray")  # 绘制手写数字图片
plt.title(f"真实标签: {label}, 预测结果: {pred}") # 设置图像标题
plt.axis("off")                                   # 隐藏坐标轴
plt.show()                                        # 展示图像

# 打印预测结果
print(f"第 {idx} 张：真实标签是 {label}，模型预测结果是 {pred}")
