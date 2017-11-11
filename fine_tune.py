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
import sys
import shutil
import argparse
import logging
from keras_generic_utils import Progbar
from util import save_checkpoint, load_checkpoint

from constant import ROOT_PATH

from data_provider import dataloders, dataset_sizes, class_names, batch_nums

model_names = ['resnet101', 'densenet', 'resnet152']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', 
                        choices=model_names, help='model architecture: ' + 
                        ' | '.join(model_names) + ' (default: resnet18)')
    args = parser.parse_args()
    return args

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

'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', modeldir='/tmp'):
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    newfile = os.path.join(modeldir, filename)
    torch.save(state, newfile)
    print ('model saved at {}'.format(newfile))
    if is_best:
        shutil.copyfile(newfile, os.path.join(modeldir,'model_best.pth.tar'))
'''


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

def train_model(model, criterion, optimizer, scheduler, num_epochs, modeldir):
    since = time.time()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    best_model_wts = model.state_dict()
    best_prec1 = 0
    train_loader = dataloders['train']
    train_batch_num = batch_nums['train']

    stop = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train() # Set model to training mode
        progbar = Progbar(train_batch_num)

        # Iterate over data.
        for batch_index, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            target = labels.cuda(async=True)
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = torch.nn.DataParallel(model)(inputs)
            # outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            prec1, prec5 = accuracy(outputs.data, target, topk=(1, 5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            progbar.add(1, values=[("p1", top1.avg), ("p5", top5.avg), ("loss", losses.avg)])
        # end of an epoch
        print()
        print('Train Loss epoch {epoch} {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               epoch=epoch, loss=losses, top1=top1, top5=top5))
            
        # evaluate on validation set
        prec1 = validate(dataloders['val'], model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
       
        stop += 1

        # deep copy the model
        if is_best:
            best_model_wts = model.state_dict()
            print ('better model obtained at epoch {epoch}'.format(epoch=epoch))
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                    }, is_best, filename='checkpoint_epoch{epoch}.pth.tar'.format(epoch=epoch), modeldir=modeldir)
            stop = 0
        if(stop >= 10):
            print("Early stop happend at {}\n".format(epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_prec1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


use_gpu = torch.cuda.is_available()
print ('use gpu? {}'.format(use_gpu))


def main(argv=None):

    args = parse_args()
    num_epochs = 100
    modeldir = os.path.join(ROOT_PATH, 'pigtrain', 'pytorch', args.arch)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if args.arch == 'resnet101':
        model_ft = models.resnet101(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif args.arch == 'resnet152':
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif args.arch == 'densenet':
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, len(class_names))


    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    '''
    val_acc = validate(dataloders['val'], model_ft, criterion)
    print ('val hit@1 {acc:.4f}'.format(acc=val_acc))
    print ('-'*10)
    '''
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, modeldir)


if __name__ == '__main__':
    sys.exit(main())

