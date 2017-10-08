from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models
import time
import os
import shutil
import logging
from keras_generic_utils import Progbar

from data_provider import dataloders, dataset_sizes, class_names, batch_nums


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', modeldir='/tmp'):
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    newfile = os.path.join(modeldir, filename)
    torch.save(state, newfile)
    print ('model saved at {}'.format(newfile))
    if is_best:
        shutil.copyfile(newfile, os.path.join(modeldir,'model_best.pth.tar'))



def validate(val_loader, model, criterion):
    print_freq = 10
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        #input_var = torch.autograd.Variable(input, volatile=True)
        #target_var = torch.autograd.Variable(target, volatile=True)
        if use_gpu:
            input_var = Variable(inputs.cuda())
            target_var = Variable(target.cuda())
        else:
            input_var, target_var = Variable(inputs), Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_prec1 = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']: #, 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            progbar = Progbar(batch_nums[phase])

            running_loss = 0.0
            running_corrects = 0
            num_seen = 0
            # Iterate over data.
            for batch_id, data in enumerate(dataloders[phase]):
                # get the inputs
                inputs, labels = data
                target = labels.cuda(async=True)
                num_seen += len(labels)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                prec1, prec5 = accuracy(outputs.data, target, topk=(1, 5))
                losses.update(loss.data[0], inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                progbar.add(1, values=[("p1", top1.avg), ("p5", top5.avg), ("loss", losses.avg)])

            #bar.finish()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print ()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # evaluate on validation set
            prec1 = validate(dataloders['val'], model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
           
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    }, is_best, filename='checkpoint_epoch{}.pth.tar'.format(epoch))


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print ('better model obtained at epoch {}'.format(epoch))
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    }, 1, filename='checkpoint_epoch{}.pth.tar'.format(epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


use_gpu = torch.cuda.is_available()
print ('use gpu? {}'.format(use_gpu))

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

arch = 'resnet'
arch = 'densenet'
#model_ft = models.resnet18(pretrained=True)
model_ft = models.densenet121(pretrained=True)
if arch == 'resnet':
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
else:
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, len(class_names))


if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

val_acc = validate(dataloders['val'], model_ft, criterion)
print ('val acc: {}'.format(val_acc))

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

