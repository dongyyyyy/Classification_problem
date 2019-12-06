import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class model(nn.Module):
    def __init__(self,in_channel=3,out_channel=6): # 128 X 128
        super(model,self).__init__()
        layer = []
        layer.append(nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=3,stride=1,padding=1))
        layer.append(nn.LeakyReLU())
        layer.append(nn.BatchNorm2d(64))
        layer.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layer.append(nn.LeakyReLU())
        layer.append(nn.BatchNorm2d(64))
        layer.append(nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv_block1 = nn.Sequential(*layer) # 64 X 64

        layer = []
        layer.append(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1))
        layer.append(nn.LeakyReLU())
        layer.append(nn.BatchNorm2d(128))
        layer.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        layer.append(nn.LeakyReLU())
        layer.append(nn.BatchNorm2d(128))
        layer.append(nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv_block2 = nn.Sequential(*layer) # 32 X 32

        layer = []
        layer.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        layer.append(nn.LeakyReLU())
        layer.append(nn.BatchNorm2d(256))
        for i in range(0,2):
            layer.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
            layer.append(nn.LeakyReLU())
            layer.append(nn.BatchNorm2d(256))
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block3 = nn.Sequential(*layer) # 16 X 16

        layer = []
        layer.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        layer.append(nn.LeakyReLU())
        layer.append(nn.BatchNorm2d(512))
        for i in range(0, 2):
            layer.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
            layer.append(nn.LeakyReLU())
            layer.append(nn.BatchNorm2d(512))
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block4 = nn.Sequential(*layer)# 8 X 8

        layer = []
        layer.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        layer.append(nn.LeakyReLU())
        layer.append(nn.BatchNorm2d(512))
        for i in range(0, 2):
            layer.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
            layer.append(nn.LeakyReLU())
            layer.append(nn.BatchNorm2d(512))
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block5 = nn.Sequential(*layer) # 4 X 4
        '''
        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        '''
        self.FC1 = nn.Linear(in_features = 4*4*512,out_features=4096)
        self.Activation = nn.LeakyReLU()
        self.FC2 = nn.Linear(in_features=4096, out_features=4096)
        self.Activation = nn.LeakyReLU()
        self.FC3 = nn.Linear(in_features=4096, out_features=out_channel)
        self.output = nn.Softmax(dim=1)
    def forward(self,input):
        x = self.conv_block1(input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.view(x.size(0),-1) # Flatten
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        out = x

        return out
