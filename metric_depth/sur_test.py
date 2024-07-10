import os
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10000, 1000)  # 输入特征维度为10，输出维度为50
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 1)  # 输出一个数值

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':

    torch.cuda.set_device(1)

    # 实例化模型
    model = SimpleNet().cuda()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 模拟数据
    # 假设我们有100个数据点，每个点10个特征
    inputs = torch.randn(100, 10000).cuda()
    targets = torch.randn(100, 1).cuda()  # 假设目标也是100个数值



    # 训练模型
    for epoch in range(100):  # 训练100轮
        optimizer.zero_grad()  # 清除之前的梯度
        output_total = list()
        # loss = {}
        total_loss = 0
        for i in range(inputs.shape[0]):  # 对每个样本逐个处理

            output = model(inputs[i].unsqueeze(0))  # 处理单个样本，升维以匹配模型输入
            output_total.append(output)
            loss = criterion(output, targets[i].unsqueeze(0))  # 计算损失
            total_loss += loss
        total_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数


        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {total_loss / inputs.shape[0]}')

    # 预测
    with torch.no_grad():  # 在预测时不需要计算梯度
        new_inputs = torch.randn(5, 10000).cuda()  # 假设有5个新的数据点
        predictions = model(new_inputs)
        print(predictions)
