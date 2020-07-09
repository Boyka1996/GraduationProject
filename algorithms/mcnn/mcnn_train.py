#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 上午11:28
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : mcnn_train.py
# @Software: PyCharm

import torch
import torch.nn as nn
from mcnn_model import MCNNModel
from torch import optim
from torchvision.transforms import transforms

from dataset import ShanghaiTechA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    # transforms.RandomCrop(300),
    transforms.ToTensor(),
    transforms.Normalize((0.452016860247, 0.447249650955, 0.431981861591),
                         (0.23242045939, 0.224925786257, 0.221840232611))
])
train_set = ShanghaiTechA('/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/', transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)

valid_set = ShanghaiTechA('/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/test_data/', transform)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=4, shuffle=True, num_workers=4)
data_loaders = {'train': train_loader, 'valid': valid_loader}
data_set_sizes = {'train': len(train_set), 'valid': len(valid_set)}
print(data_set_sizes)
'''train model'''


def train():
    # prepare model
    model = MCNNModel()
    best_model_wts = model.state_dict()
    # print(best_model_wts)
    best_mae = float('inf')
    best_mse = float('inf')
    model.to(device)
    start_epoch = 1
    end_epoch = 15
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

    # train

    for epoch in range(start_epoch, end_epoch):
        print('Epoch {}/{}'.format(epoch, end_epoch - start_epoch + 1))

        for phase in ['train', 'valid']:
            print(phase)
            if phase == 'train':
                model.train(mode=True)
            else:
                model.eval()
            running_loss = 0.0
            running_mae = 0.0
            running_mse = 0.0

            for images, gts in data_loaders[phase]:
                optimizer.zero_grad()

                images, gts = images.to(device), gts.to(device)

                if phase == 'train':
                    output = model(images)
                    # plt.imshow(output.to('cpu').detach().numpy()[0][0])
                    # plt.show()
                    loss = criterion(gts, output)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        output = model(images)
                        # plt.imshow(output.to('cpu').detach().numpy()[0][0])
                        # plt.show()
                        loss = criterion(gts, output)
                running_loss += loss
                running_mae += torch.sum(torch.abs(output - gts))
                running_mse += torch.sum(pow(output - gts, 2))
                running_mse += 0
            epoch_loss = running_loss.item() / data_set_sizes[phase]
            epoch_mae = running_mae.item() / data_set_sizes[phase]
            epoch_mse = pow(running_mse / data_set_sizes[phase], 0.5).item()
            print('{} Loss: {} MAE: {} MSE: {}'.format(phase, epoch_loss, epoch_mae, epoch_mse))

            if phase == 'valid' and epoch_mae < best_mae:
                best_mae = epoch_mae
                best_mse = epoch_mse
                best_model_wts = model.state_dict()
                # print(best_model_wts)

    print('Best val mae: {:.4f}'.format(best_mae))
    print('Best val mse: {:.4f}'.format(best_mse))

    model.load_state_dict(best_model_wts)
    torch.save(model, "./best.pth")
    return model


'''run'''
if __name__ == '__main__':
    train()
