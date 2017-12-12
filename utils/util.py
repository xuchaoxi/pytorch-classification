from __future__ import print_function, division

import torch
import shutil
import os

DEFAULT_MODEL_DIR = '/home/xcx/VisualSearch/checkpoint'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint=DEFAULT_MODEL_DIR):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    newfile = os.path.join(checkpoint, filename)
    torch.save(state, newfile)
    print ('model saved at {}'.format(newfile))
    if is_best:
        shutil.copyfile(newfile, os.path.join(checkpoint,'model_best.pth.tar'))


def load_checkpoint(filename='model_best.pth.tar', modeldir=DEFAULT_MODEL_DIR):
    checkpoint = torch.load(os.path.join(modeldir,filename))
    return checkpoint


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

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
