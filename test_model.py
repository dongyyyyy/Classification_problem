import csv
import numpy as np
import torch
from torchsummary import summary
import sys
from torch.utils.data import DataLoader
from DataLoader import *
from models import *
from torch.autograd import Variable
import os


import numpy as np
import torch
from torchsummary import summary
import sys
from torch.utils.data import DataLoader
from DataLoader import *
from models import *
from torch.autograd import Variable
import os

if __name__ == '__main__':
    batch_size = 12

    cuda = torch.cuda.is_available()

    model = model()

    if cuda:
        model = model.cuda()
    #loss_model = nn.

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataloader = DataLoader(
        FaceDataset(train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    model.load_state_dict(torch.load("saved_models/best_model.pth"))
    f = open('output1.csv','w',encoding='utf-8',newline='')
    wr = csv.writer(f)
    wr.writerow(["prediction"])
    for i,data in enumerate(dataloader):
        imgs = data["img"]
        #label_onehot =
        imgs = Variable(imgs.type(Tensor))

        pred = model(imgs)
        pred = F.softmax(pred,dim=0) # softmax
        pred = torch.argmax(pred,dim=1)
        for j in range(pred.size(0)):
            result = pred[j].cpu().numpy()
            print(result)
            wr.writerow([result])


        '''
        predict = torch.sigmoid(pred)
        predict = torch.argmax(predict, dim=1)
        predict = predict.tolist()
        labels = label.tolist()
        for j in range(imgs.size(0)):
            if predict[j] == labels[j]:"
                # sys.stdout.write("Compare : %d %d\n"%(predict[i], labels[i]))
                correct += 1
        '''
        #path = data["path"]
    f.close()