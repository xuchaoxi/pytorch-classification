from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import time
import os
import shutil

from data_provider import dataloders, dataset_sizes
from util import load_checkpoint

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
use_gpu = torch.cuda.is_available()

resume = '/tmp/model_best.pth.tar'
checkpoint = load_checkpoint(resume)
start_epoch = checkpoint['epoch']
best_acc = checkpoint['best_acc']
model.load_state_dict(checkpoint['state_dict'])

if use_gpu:
    model = model.cuda()
model.train(False)  

#optimizer.load_state_dict(checkpoint['optimizer'])
print("=> loaded checkpoint '{}' (epoch {}, val acc {})".format(resume, checkpoint['epoch'], best_acc))


def test(model):
    running_corrects = 0 
    for i, data in enumerate(dataloders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects / dataset_sizes['val']
    print (acc)

test(model)
