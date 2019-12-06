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
    os.makedirs("saved_models",exist_ok=True)
    learning_rate = 0.0001
    b1 = 0.5
    b2 = 0.999
    start_epochs = 0
    epochs = 40
    batch_size = 12

    cuda = torch.cuda.is_available()

    model = model()

    if cuda:
        model = model.cuda()
    summary(model,input_size=(3,128,128))
    #loss_model = nn.

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(b1,b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataloader = DataLoader(
        FaceDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    if start_epochs != 0:  # 처음부터 학습이 아닐 경우에는 saved_models에서 해당 시작 위치에 해당하는 checkpoint 정보 가져오기
        # Load pretrained models
        model.load_state_dict(torch.load("saved_models/best_model.pth"))
        # generator.load_state_dict(torch.load("saved_models/generator_%d.pth"%opt.epoch))
        # discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"%opt.epoch))
    avg_loss = 0.0
    for epoch in range(start_epochs, epochs):
        total = 0
        #correct = 0
        current_avg_loss = 0.0
        time = 0
        for i,data in enumerate(dataloader):
            imgs = data["img"]
            label = data["label"]
            #label_onehot =
            imgs = Variable(imgs.type(Tensor))
            label = label.cuda()
            optimizer.zero_grad()

            pred = model(imgs)
            #pred = F.softmax(pred,dim=0) # softmax
            loss = F.cross_entropy(pred,label)
            current_avg_loss += loss.item()
            time += 1
            total += imgs.size(0)

            loss.backward()
            optimizer.step()
            '''
            predict = torch.sigmoid(pred)
            predict = torch.argmax(predict, dim=1)
            predict = predict.tolist()
            labels = label.tolist()
            for j in range(imgs.size(0)):
                if predict[j] == labels[j]:
                    # sys.stdout.write("Compare : %d %d\n"%(predict[i], labels[i]))
                    correct += 1
            '''
            #path = data["path"]

            #print(path)
            if i%10 == 0:
                sys.stdout.write("[Epoch %d/%d] [Batch %d/%d] [loss: %f]\n" % (epoch, epochs, i, len(dataloader), loss.item()))
                #pred = torch.argmax(pred,dim=1)
                #print("pred : ", pred)
                #print("label : ", label)
                #sys.stdout.write("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [Acc: %f]"%(epoch,epochs,i,len(dataloader),loss.item(),100*correct/total))
        current_avg_loss = current_avg_loss/time
        if epoch == 0:
            avg_loss = current_avg_loss
            sys.stdout.write("current_avg_loss : %f / avg_loss : %f\n" % (current_avg_loss, avg_loss))
            torch.save(model.state_dict(),"saved_models/best_model.pth")
        else:
            sys.stdout.write("current_avg_loss : %f / avg_loss : %f\n" % (current_avg_loss, avg_loss))
            if avg_loss > current_avg_loss:
                avg_loss = current_avg_loss
                torch.save(model.state_dict(),"saved_models/best_model.pth")
