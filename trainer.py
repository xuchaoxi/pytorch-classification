from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
import models
import time
import os
import sys
import shutil
import argparse
import logging
from utils.keras_generic_utils import Progbar
from utils import save_checkpoint, load_checkpoint, AverageMeter, Logger, accuracy, mkdir_p

from constant import ROOT_PATH
from data_provider import dataloders, class_names, batch_nums

logprint = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
        datefmt='%d %b %H:%M:%S')
logprint.setLevel(logging.INFO)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
print(model_names)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_gpu = torch.cuda.is_available()
print ('use gpu? {}'.format(use_gpu))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', 
                        choices=model_names, help='model architecture: ' + 
                        ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='overwrite checkpoint ')
    args = parser.parse_args()
    return args


def validate(val_loader, model, criterion, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    print()
    progbar = Progbar(batch_nums['val'])
    for i, (inputs, target) in enumerate(val_loader):
        if use_gpu:
            input_var = Variable(inputs.cuda(), volatile=True)
            target_var = Variable(target.cuda())
        else:
            input_var, target_var = Variable(inputs), Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.evaluate:
            logger.append([losses.avg, top1.avg])
        else:
            logger.append([None, None, losses.avg, None, top1.avg])
        progbar.add(1, values=[("p1", top1.avg), ("p5", top5.avg), ("loss", losses.avg)])

        '''
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
        '''
    print()
    print(' * Current Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def train_model(model, criterion, optimizer, scheduler):
    since = time.time()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    best_model_wts = model.state_dict()
    best_prec1 = 0
    train_loader = dataloders['train']
    train_batch_num = batch_nums['train']
    start_epoch = args.start_epoch

    stop = 0

    log_file = os.path.join(args.checkpoint, 'log.txt')

    title = 'cifar-10-' + args.arch
    if args.resume:
        print('==> Resuming from checkpoint...')
        print(args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_prec1 = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(log_file, title=title, resume=True)
    else:
        if os.path.exists(log_file) and not args.overwrite:
            logprint.info('%s exists. skip', log_file)
            return 0
        logger = Logger(log_file, title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    for epoch in range(start_epoch, args.epochs):

        scheduler.step()
        model.train() # Set model to training mode

        print('Epoch [{} | {}] LR: {}'.format(epoch, args.epochs - 1, scheduler.get_lr()[0]))
        print('-' * 10)
        progbar = Progbar(train_batch_num)

        # Iterate over data.
        for batch_index, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda(async=True))
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            logger.append([scheduler.get_lr()[0], losses.avg, None, top1.avg, None])
            progbar.add(1, values=[("p1", top1.avg), ("p5", top5.avg), ("loss", losses.avg)])
        # end of an epoch
        print()
        print('Train Loss epoch {epoch} {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               epoch=epoch, loss=losses, top1=top1, top5=top5))
            
        # evaluate on validation set
        prec1 = validate(dataloders['val'], model, criterion, logger)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
       
        stop += 1
        print(' * Best Prec@1 {top1:.3f}'.format(top1=best_prec1))

        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                }, is_best, filename='checkpoint_epoch{epoch}.pth.tar'.format(epoch=epoch), checkpoint=args.checkpoint)
        # deep copy the model
        if is_best:
            best_model_wts = model.state_dict()
            print ('better model obtained at epoch {epoch}'.format(epoch=epoch))
            stop = 0
        if(stop >= 20):
            print("\nEarly stop happend at {}\n".format(epoch))
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_prec1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def main(argv=None):

    global args
    args = parse_args()

    args.checkpoint = os.path.join(args.checkpoint, args.arch+'_lr_'+str(args.lr))

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=len(class_names))

    if use_gpu:
        model = model.cuda()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        print('\nEvaluating only ...')
        if args.resume:
            print('==> Resuming from checkpoint...')
            print(args.resume)
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            args.checkpoint = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print ('best_prec1 hit@1 {acc:.4f}'.format(acc=best_prec1))

        title = 'cifar-10-' + args.arch
        logger = Logger(os.path.join(args.checkpoint, 'evaluate-log.txt'), title=title)
        logger.set_names(['Valid Loss', 'Valid Acc.'])

        val_acc = validate(dataloders['val'], model, criterion, logger)
        print ('val hit@1 {acc:.4f}'.format(acc=val_acc))
        print ('-'*10)
        return

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler)


if __name__ == '__main__':
    sys.exit(main())

