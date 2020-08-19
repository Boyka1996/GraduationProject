#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 下午3:46
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


# from torch import nn


class ConvModule(nn.Module):
    # def _init_(self):
    def __init__(self):
        super(ConvModule, self).__init__()
        # super(ConvModule, self)._init_()
        # 定义三层卷积//
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(128, 3)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2D(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2D(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2D(128),
        #     nn.ReLU(inplace=True)
        # )
        # # 输出层，将通道数变为分类数量
        # self.fc = nn.Linear(128, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(x))
        out = F.max_pool2d(out)
        out = self.fc(out)
        return out

    # def forward(self, X):
    #     # 图片先经过三层卷积，输出维度（batch_size,C_out,H,W)
    #     out = self.conv(X)
    #     # 使用平均池化层将图片的大小变为1*1
    #     out = F.avg_pool2d(out, 26)
    #     # 将张量out从shape batch*32*1*1变为 batch*32
    #     out = out.squeeze()
    #     # 输入到全连接层将输出的维度变为10
    #     out = self.fc(out)
    #     return out


# 训练函数
def train_epoch(net, data_loader, device):
    net.train()  # 指定当前为训练模式
    train_batch_num = len(data_loader)  # 记录共有多少个batch
    total_loss = 0  # 记录loss
    correct = 0
    sample_num = 0

    # 遍历每个batch进行xunlian
    for batch_idx, (data, target) in enumerate(data_loader):
        # 将图片放入指定的divice中
        data = data.to(device).float()
        # 将图片标签放入指定的devide中
        target = target.to(device).long()
        # 将当前梯度清零
        optimizer.zero_grad()
        # 使用模型计算出结果
        output = net(data)
        # 计算损失
        loss = criterion(output, target)
        # 进行反向传播
        loss.backward()
        optimizer.step()

        # 累加loss
        total_loss += loss.item()
        # 找出每个样本值最大的idx，即代表预测此图片属于哪个类别
        prediction = torch.argmax(output, 1)
        # 统计预测正确的类别数量
        correct += (prediction == target).sum().item()
        # 累加当前的样本总数
        sample_num += len(prediction)
    # 计算平均的loss和准确率
    loss = total_loss / train_batch_num
    acc = correct / sample_num
    return loss, acc


# 测试函数
def test_epoch(net, data_loader, device):
    net.eval()  # 指定当前为训练模式
    test_batch_num = len(data_loader)  # 记录共有多少个batch
    total_loss = 0  # 记录loss
    correct = 0
    sample_num = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # 将图片放入指定的divice中
            data = data.to(device).float()
            # 将图片标签放入指定的devide中
            target = target.to(device).long()
            # 使用模型计算出结果
            output = net(data)
            # 计算损失
            loss = criterion(output, target)
            # 累加loss
            total_loss += loss.item()
            # 找出每个样本值最大的idx，即代表预测此图片属于哪个类别
            prediction = torch.argmax(output, 1)
            # 统计预测正确的类别数量
            correct += (prediction == target).sum().item()
            # 累加当前的样本总数
            sample_num += len(prediction)
        # 计算平均的loss和准确率
    loss = total_loss / test_batch_num
    acc = correct / sample_num
    return loss, acc


classes = ('bus', 'car', 'truck')
num_classes = 3  # 共3类
epochs = 100
lr = 0.001
batch_size = 512
device = torch.device("cuda:0")  # 若使用cpu则填写cpu

# 读取数据集
datasets = torchvision.datasets.ImageFolder(root="/home/chase/datasets/车辆分类数据集",
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize((200, 100))
                                            ]))

trainset, testset = train_test_split(datasets.imgs, test_size=0.2)
# 查看数据
# print(datasets.imgs)
# 查看标签
# print(datasets.classes)

# 批量加载数据集
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size)

net = ConvModule()
criterion = nn.CrossEntropyLoss()
print(list(net.parameters()))
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# 模型训练并验证
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(net, data_loader=trainloader, device=device)
    test_loss, test_acc = test_epoch(net, data_loader=testloader, device=device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    print(f"epoch:{epoch}\t train_loss:{train_loss:.4f} \t"
          f"train_acc:{train_acc} \t"
          f"test_loss:{test_loss:.4f} \t test_acc:{test_acc}")
