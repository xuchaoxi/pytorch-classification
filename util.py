from __future__ import print_function, division

import torch
import os


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', modeldir='/tmp'):
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    newfile = os.path.join(modeldir, filename)
    torch.save(state, newfile)
    print ('model saved at {}'.format(newfile))
    if is_best:
        shutil.copyfile(newfile, os.path.join(modeldir,'model_best.pth.tar'))


def load_checkpoint(filename='model_best.pth.tar', modeldir='/tmp'):
    checkpoint = torch.load(os.path.join(modeldir,filename))
    return checkpoint

