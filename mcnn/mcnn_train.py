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
from mcnn_model import MCNNModel
from torch import optim
from torchvision.transforms import transforms

from dataset import ShanghaiTechA

# from mcnn import MCNNModel
#
# from dataset import ShanghaiTechA
# from . import dataset,mcnn_model
# from mcnn_model import mcnn_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    # transforms.RandomCrop(300),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor()
# ])

train_set = ShanghaiTechA('/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/', transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)

test_set = ShanghaiTechA('/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/test_data/', transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=4)
'''train model'''


def train():
    # prepare model
    model = MCNNModel()
    model.to(device)
    start_epoch = 1
    end_epoch = 10
    my_filter=torch.nn.MaxPool2d(kernel_size=4, stride=4).to(device)
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
            imgs, gt = imgs.to(device), gt.to(device)
            output = model(imgs)
            loss = criterion(my_filter(gt), output)
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
