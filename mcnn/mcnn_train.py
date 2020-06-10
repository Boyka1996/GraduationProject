#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 上午11:28
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : mcnn_train.py
# @Software: PyCharm
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from torch import optim
from torchvision.transforms import transforms

from .mcnn_model import MCNNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=4)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
'''train model'''


def train():
    # prepare model
    model = MCNNModel()
    model.to(device)
    start_epoch = 1
    end_epoch = 10
    criterion = nn.MSELoss()
    # prepare optimizer
    learning_rate_idx = 0
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9,
                          weight_decay=0.3)
    # train
    for epoch in range(start_epoch, end_epoch + 1):
        # --train epoch
        for batch_idx, samples in enumerate(train_loader):
            optimizer.zero_grad()
            imgs, gt = samples
            output = model(imgs)
            loss = criterion(gt, output)
            loss.backward()
            optimizer.step()
        # --save model
        if (epoch % 2 == 0) or (epoch == end_epoch):
            state_dict = {'epoch': epoch,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            save_path = os.path.join('/home/chase/projects/GraduationProject/mcnn/checkpoints', 'epoch_%s.pth' % epoch)
            torch.save(state_dict, save_path)


'''run'''
if __name__ == '__main__':
    train()
