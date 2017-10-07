from __future__ import print_function, division

import torch
from torchvision import datasets, transforms
import os

data_transforms = {
    'train': transforms.Compose([
        #transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'

rename = {'train':'ai_challenger_scene_train_20170904', 'val':'ai_challenger_scene_val_20170908'}
data_dir = os.path.join(os.environ['HOME'], 'VisualSearch')
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, rename[x], 'ImageData'),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=(x=='train'), num_workers=2)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print (class_names, dataset_sizes)
print (len(class_names))

