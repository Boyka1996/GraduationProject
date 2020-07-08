#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 下午10:52
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from mcnn_model import MCNNModel
from torchvision.transforms import transforms
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MCNNModel()
    model=torch.load('/home/chase/projects/GraduationProject/mcnn/best.pth')
    # model.load_state_dict(torch.load('/home/chase/projects/GraduationProject/mcnn/best.pth'))
    model.eval()
    model.to(device)
    image = Image.open(
        '/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/test_data/images/IMG_1.jpg').convert("RGB")
    transform = transforms.Compose([
        # transforms.RandomCrop(300),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image=transform(image)
    image=torch.unsqueeze(image,0)
    image = image.to(device)
    print(image.shape)
    result = model(image).to('cpu').detach()
    result=torch.squeeze(result)
    result=result.numpy()
    print(result)
    print(np.sum(result))

    plt.imshow(result)
    plt.show()
