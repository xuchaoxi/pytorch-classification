from __future__ import print_function, division

import torch
import os

DEFAULT_MODEL_DIR = '/tmp/pytorch'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', modeldir=DEFAULT_MODEL_DIR):
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    newfile = os.path.join(modeldir, filename)
    torch.save(state, newfile)
    print ('model saved at {}'.format(newfile))
    if is_best:
        shutil.copyfile(newfile, os.path.join(modeldir,'model_best.pth.tar'))


def load_checkpoint(filename='model_best.pth.tar', modeldir=DEFAULT_MODEL_DIR):
    checkpoint = torch.load(os.path.join(modeldir,filename))
    return checkpoint

