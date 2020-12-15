from __future__ import print_function, absolute_import

import os
import csv
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from os import listdir
from os.path import isfile, join

import torch
import torch.utils.data as data

from scripts.utils.osutils import *
from scripts.utils.imutils import *
from scripts.utils.transforms import *
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BIH(data.Dataset):
    def __init__(self,train,config=None, sample=[],gan_norm=False):

        self.train = []
        self.anno = []
        self.mask = []
        self.wm = []
        self.input_size = config.input_size
        self.normalized_input = config.normalized_input
        self.base_folder = config.base_dir +'/' + config.data
        self.dataset = config.data

        if config == None:
            self.data_augumentation = False
        else:
            self.data_augumentation = config.data_augumentation

        self.istrain = False if train.find('train') == -1 else True
        self.sample = sample
        self.gan_norm = gan_norm
        mypath = join(self.base_folder,self.dataset+'_'+train+'.txt')

        with open(mypath) as f:
            # here we get the filenames 
            file_names = [ im.strip() for im in f.readlines() ]

        if config.limited_dataset > 0:
            xtrain = sorted(list(set([ file_name.split('-')[0] for file_name in file_names ])))
            tmp = []
            for x in xtrain:
                tmp.append([y for y in file_names if x in y][0])

            file_names = tmp
        else:
            file_names = file_names

        for file_name in file_names:
            self.train.append(os.path.join(self.base_folder,'images',file_name))
            self.mask.append(os.path.join(self.base_folder,'masks','_'.join(file_name.split('_')[0:2])+'.png'))
            self.anno.append(os.path.join(self.base_folder,'reals',file_name.split('_')[0]+'.jpg'))

        if len(self.sample) > 0 :
            self.train = [ self.train[i] for i in self.sample ] 
            self.mask = [ self.mask[i] for i in self.sample ] 
            self.anno = [ self.anno[i] for i in self.sample ] 

        self.trans = transforms.Compose([
                transforms.Resize((self.input_size,self.input_size)),
                transforms.ToTensor()
            ])

        print('total Dataset of '+self.dataset+' is : ', len(self.train))


    def __getitem__(self, index):
        img = Image.open(self.train[index]).convert('RGB')
        mask = Image.open(self.mask[index]).convert('L')
        anno = Image.open(self.anno[index]).convert('RGB')

        # for shadow removal and blind image harmonization, here is no ground truth wm
        # wm = Image.open(self.wm[index]).convert('RGB')

        return {"image": self.trans(img),
                "target": self.trans(anno), 
                "mask": self.trans(mask), 
                "name": self.train[index].split('/')[-1],
                "imgurl":self.train[index],
                "maskurl":self.mask[index],
                "targeturl":self.anno[index],
                }

    def __len__(self):

        return len(self.train)
