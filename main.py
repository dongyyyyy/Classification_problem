import numpy as np
import torch
from torchsummary import summary
import sys
from torch.utils.data import DataLoader
from DataLoader import *
from models import *
from torch.autograd import Variable

if __name__ == '__main__':

    learning_rate = 0.0001
    b1 = 0.5
    b2 = 0.999
    epochs = 20
    batch_size = 4

    cuda = torch.cuda.is_available()

    model = model()

    if cuda:
        model = model.cuda()
    summary(model,input_size=(3,128,128))
    loss_model = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(b1,b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataloader = DataLoader(
        FaceDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    for epoch in range(epochs):
        total = 0
        correct = 0
        for i,data in enumerate(dataloader):
            imgs = data["img"]
            label = data["label"]
            #label_onehot =
            imgs = Variable(imgs.type(Tensor))
            label = label.cuda()
            optimizer.zero_grad()

            pred = model(imgs)
            loss = loss_model(pred,label)
            total += imgs.size(0)
            #correct += (pred == label).sum().item()
            loss.backward()
            optimizer.step()
            #path = data["path"]

            #print(path)
            if i%10 == 0:
                sys.stdout.write("[Epoch %d/%d] [Batch %d/%d] [loss: %f]\n" % (epoch, epochs, i, len(dataloader), loss.item()))
                pred = torch.argmax(pred,dim=1)
                print("pred : ", pred)
                print("label : ", label)
                #sys.stdout.write("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [Acc: %f]"%(epoch,epochs,i,len(dataloader),loss.item(),100*correct/total))
