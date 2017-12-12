from __future__ import print_function, division

import torch
from torchvision import datasets, transforms
import os
import math

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "/home/xcx/code/AI_class/Machine_l/net/data"
batch_size = 64
image_datasets = {x: datasets.CIFAR10(root=data_dir, train=(x=='train'), download=False, transform=data_transforms[x])
                  for x in ['train', 'val']}

dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=(x=='train'), num_workers=2)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

batch_nums = {x: int(math.ceil(dataset_sizes[x]/batch_size)) for x in ['train', 'val']}

print('dataset_size: ', dataset_sizes)
print ('batch_size: ', batch_size, '\nbatch_nums: ', batch_nums)

class_names = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    #train_loader = dataloders['train']
    print(len(image_datasets['train']))
    for index, data in enumerate(image_datasets['train']):
        #inputs, labels = data
        print(data)
        break
