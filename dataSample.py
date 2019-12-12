import numpy as np
import torch
from torchsummary import summary
import sys
from torch.utils.data import DataLoader
from DataLoader import *
from models import *
from torch.autograd import Variable
import os
batch_size = 8

if __name__ == '__main__':
    labels = [0,0,0,0,0,0]
    dataloader = DataLoader(
            FaceDataset(),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )

    for i,data in enumerate(dataloader):
        label = data["label"]
        answer = label.cpu().numpy()
        for j in range(len(answer)):
            labels[answer[j]] += 1

    print(labels)