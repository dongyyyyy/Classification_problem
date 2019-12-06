import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class FaceDataset(data.Dataset):
    def __init__(self,
                 train=True):
        self.train = train
        self.img_path = './faces_images'

        self.train_img_path = []
        self.train_label = []

        self.test_img_path = []

        self.transforms = transforms.Compose(
            [
                transforms.Resize((128,128), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ]
        )

        if self.train:
            train_file_name = []

            with open('./train_vision.csv', 'r') as f:
                for line in f:
                    train_file_name.append(line.strip('\n'))

            for name_label in train_file_name[1:]:
                name, label = name_label.split(',')
                self.train_img_path.append(os.path.join(self.img_path, name))
                self.train_label.append(label)

        else:
            test_file_name = []

            with open('./test_vision.csv', 'r') as f:
                for line in f:
                    test_file_name.append(line.strip('\n'))

            for name in test_file_name[1:]:
                self.test_img_path.append(os.path.join(self.img_path, name))

    def __getitem__(self, index):
        if self.train:
            path = self.train_img_path[index]

            image = Image.open(path)
            label = int(self.train_label[index])-1

            image = self.transforms(image)

            return {"img":image, "label":label, "path": path}

        else:
            path = self.test_img_path[index]

            image = Image.open(path)

            image = self.transforms(image)

            return {"img":image, "path": path}

    def __len__(self):
        if self.train:
            return len(self.train_img_path)
        else:
            return len(self.test_img_path)