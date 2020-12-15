from __future__ import absolute_import

import os
import shutil
import torch 
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch.nn.functional as F

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def resize_to_match(fm,to):
    # just use interpolate
    # [1,3] = (h,w)
    return F.interpolate(fm,to.size()[-2:],mode='bilinear',align_corners=False)

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(machine,filename='checkpoint.pth.tar', snapshot=None):
    is_best = True if machine.best_acc < machine.metric else False

    if is_best:
        machine.best_acc = machine.metric

    state = {
                'epoch': machine.current_epoch + 1,
                'arch': machine.args.arch,
                'state_dict': machine.model.state_dict(),
                'best_acc': machine.best_acc,
                'optimizer' : machine.optimizer.state_dict(),
            }

    filepath = os.path.join(machine.args.checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(machine.args.checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))
       
    if is_best:
        machine.best_acc = machine.metric
        print('Saving Best Metric with PSNR:%s'%machine.best_acc)
        shutil.copyfile(filepath, os.path.join(machine.args.checkpoint, 'model_best.pth.tar'))
        


def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def adjust_learning_rate(datasets,optimizer, epoch, lr,args):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in args.schedule:
        lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # decay sigma
    for dset in datasets:
        if args.sigma_decay > 0:
            dset.dataset.sigma *=  args.sigma_decay
            dset.dataset.sigma *=  args.sigma_decay

    return lr




