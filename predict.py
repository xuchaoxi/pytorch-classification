from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import time
import os
import shutil
import argparse
from PIL import Image

#from data_provider import class_names
from util import load_checkpoint
from constant import ROOT_PATH


model_names = ['resnet101', 'densenet', 'resnet152']
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_transforms = transforms.Compose([
                        transforms.RandomSizedCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

data_dir = os.path.join(ROOT_PATH, 'pigtest', 'ImageData')
class_names = open('pig_classnames.txt').readlines().strip()
print(len(class_names), class_names)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        choices=model_names, help='model architecture: ' + 
                        ' | '.join(model_names) + ' (default: resnet18)')
    args = parser.parse_args()
    return args

use_gpu = torch.cuda.is_available()

def test(model, res_dir):

    img_ids = os.listdir(data_dir)
    res = {}
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    res_file = os.path.join(res_dir, 'submission.csv')
    m = nn.Softmax()
    fw = open(res_file, 'w')
    for i, p_id in enumerate(img_ids):
        im_data = Image.open(os.path.join(data_dir, p_id))
        im_data = im_data.convert('RGB')

        im_data = data_transforms(im_data)

        im_data.unsqueeze_(0)

        if use_gpu:
            im_data = Variable(im_data.cuda())
        else:
            im_data = Variable(im_data)

        output = model(im_data)
        output = m(output)
        max_p, index = torch.max(output.data, 1)
        '''  
        # worse method
        if max_p[0] >= 0.4:
            output.data = torch.zeros(1,30)
            output.data[0][index[0]] = 1.
        '''
        res[p_id.split('.')[0]] = output.data
        for k in range(len(class_names)):
            fw.write('{},{},{}\n'.format(p_id.split('.')[0], class_names[k], output.data[0][k]))

        if(i % 500 == 0):
            print("==========> {} pictures has been predicted. ========>".format(i))

    fw.close()
    with open('submission.csv', 'w') as f:
        for k in res:
            for i in range(len(class_names)):
                f.write('{},{},{}\n'.format(k, class_names[i],res[k][0][i]))

        #_, preds = torch.max(outputs.data, 1)

def main(rootpath=ROOT_PATH):
    args = parse_args()

    resume = os.path.join(rootpath, 'pigtest', 'pytorch', args.arch, 'model_best.pth.tar')
    res_dir = os.path.join(rootpath, 'pigtest', 'pytorch', args.arch, 'submission.csv')

    if args.arch == 'resnet101':
        model = models.resnet101(pretrain=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_fts, len(class_names))
    elif args.arch == 'resnet152':
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    elif args.arch == 'denset121':
        model = models.desnet121(pretrained=False)
        num_ftrs = model.classifer.in_features
        model.classifier = nn.Linear(num_ftrs, len(class_names))

    checkpoint = load_checkpoint(resume)

    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])

    #optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {}, val acc {})".format(resume, checkpoint['epoch'], best_acc))
        
    if use_gpu:
        model = model.cuda()
    #model.train(False)  

    test(model, res_dir)

if __name__ == '__main__':
    main()
