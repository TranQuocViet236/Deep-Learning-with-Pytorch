import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

# %matplotlib inline
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from tqdm import tqdm


class ImageTransform():
  def __init__(self, resize, mean, seed):
    self.data_transform = {
        'train': transforms.Compose([
                                     transforms.RandomResizedCrop(resize, scale= (0.5, 1.0)),
                                     transforms.RandomHorizontalFlip(0.7),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
                                     transforms.Resize(resize),
                                     transforms.CenterCrop(resize),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
        ])
    }

  def __call__(self, img, phase = 'train'):
    return self.data_transform[phase](img)

def make_datapath_list(phase = 'train'):
  root_path = "data/hymenoptera_data/"
  target_path = osp.join(root_path + phase + "/**/*.jpg")
  # print(target_path)

  path_list = []

  for path in glob.glob(target_path):
    path_list.append(path)

  return path_list


def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_param_name_1 = ["features"]
    update_param_name_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
        if name in update_param_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)

        elif name in update_param_name_2:
            param.requires_grad = True
            params_to_update_2.append(param)

        elif name in update_param_name_3:
            param.requires_grad = True
            params_to_update_3.append(param)

        else:
            param.requires_grad = False

        return params_to_update_1, params_to_update_2, params_to_update_3


class MyDataSet(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        label = img_path.split('/')[-2]
        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transformed, label

train_list = make_datapath_list('train')
val_list = make_datapath_list('val')

num_epochs = 1
size = 224
mean = (0.485, 0.456, 0.406)
std= (0.229, 0.224, 0.225)
train_dataset = MyDataSet(train_list, transform= ImageTransform(size, mean, std), phase= 'train')
val_dataset = MyDataSet(val_list, transform= ImageTransform(size, mean, std), phase= 'val')

#data loader
save_path = "./weigh_fine_tuning.pth"
batch_size = 4
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

#network
use_pretrained = True
net = models.vgg16(pretrained = use_pretrained)
# print(net)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

#loss
criterior = nn.CrossEntropyLoss()

params1, params2, params3 = params_to_update(net)


from torch.optim import optimizer
# optimizer = optim.SGD(params=params, lr=0.001, momentum=0.9)
optimizer = optim.SGD([
                       {'params': params1, 'lr': 1e-4},
                       {'params': params2, 'lr': 5e-4},
                       {'params': params3, 'lr':1e-3},
], momentum=0.9)


def train_model(net, dataloader_dict, criterior, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # move network to device(GPU/CPU)
    net.to(device)

    torch.backends.cudnn.benchmark = True  # Tăng tốc khả năng tính toán của network

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                print(f'Type of inputss: {type(inputs)}')
                print(f'Type of labels: {type(labels)}')
                # move inputs, labels to device(GPU/CPU)
                inputs = inputs.to(device)

                labels_unique = set(labels)
                keys = {key: value for key, value in zip(labels_unique, range(len(labels_unique)))}
                labels_onehot = torch.zeros(size=(len(labels), len(keys)))
                for idx, label in enumerate(labels_onehot):
                    labels_onehot[idx][keys[label]] = 1
                labels_onehot = labels.to(device)
                # labels = torch.tensor(labels).to(device)
                # set gradient of optimizer to be zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

    torch.save(net.state_dict()
               , save_path
               )

train_model(net, dataloader_dict, criterior, optimizer, num_epochs)