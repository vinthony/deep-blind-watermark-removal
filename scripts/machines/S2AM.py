import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from progress.bar import Bar
import json
import numpy as np
from tensorboardX import SummaryWriter
from scripts.utils.evaluation import accuracy, AverageMeter, final_preds
from scripts.utils.osutils import mkdir_p, isfile, isdir, join
from scripts.utils.parallel import DataParallelModel, DataParallelCriterion
import pytorch_ssim as pytorch_ssim
import torch.optim
import sys,shutil,os
import time
import scripts.models as archs
from math import log10
from torch.autograd import Variable
from scripts.utils.losses import VGGLoss
from scripts.utils.imutils import im_to_numpy

import skimage.io
from skimage.measure import compare_psnr,compare_ssim


class S2AM(object):
    def __init__(self, datasets =(None,None), models = None, args = None, **kwargs):
        super(S2AM, self).__init__()
        
        self.args = args
        
        # create model
        print("==> creating model ")
        self.model = archs.__dict__[self.args.arch]()
        print("==> creating model [Finish]")
       
        self.train_loader, self.val_loader = datasets
        self.loss = torch.nn.MSELoss()
        
        self.title = '_'+args.machine + '_' + args.data + '_' + args.arch
        self.args.checkpoint = args.checkpoint + self.title
        self.device = torch.device('cuda')
         # create checkpoint dir
        if not isdir(self.args.checkpoint):
            mkdir_p(self.args.checkpoint)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            lr=args.lr,
                            betas=(args.beta1,args.beta2),
                            weight_decay=args.weight_decay)  
        
        if not self.args.evaluate:
            self.writer = SummaryWriter(self.args.checkpoint+'/'+'ckpt')
        
        self.best_acc = 0
        self.is_best = False
        self.current_epoch = 0
        self.hl = 1
        self.metric = -100000
        self.count_gpu = len(range(torch.cuda.device_count()))

        if self.args.style_loss > 0:
            # init perception loss
            self.vggloss = VGGLoss(self.args.sltype).to(self.device)

        if self.count_gpu > 1 : # multiple
            # self.model = DataParallelModel(self.model, device_ids=range(torch.cuda.device_count()))
            # self.loss = DataParallelCriterion(self.loss, device_ids=range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.model.to(self.device)
        self.loss.to(self.device)

        print('==> Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))
        print('==> Total devices: %d' % (torch.cuda.device_count()))
        print('==> Current Checkpoint: %s' % (self.args.checkpoint))


        if self.args.resume != '':
            self.resume(self.args.resume)


    def train(self,epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossvgg = AverageMeter()
        
        # switch to train mode
        self.model.train()
        end = time.time()

        bar = Bar('Processing', max=len(self.train_loader)*self.hl)
        for _ in range(self.hl):
            for i, batches in enumerate(self.train_loader):
                # measure data loading time
                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask =batches['mask'].to(self.device)
                current_index = len(self.train_loader) * epoch + i

                feeded = torch.cat([inputs,mask],dim=1)
                feeded = feeded.to(self.device)

                output = self.model(feeded)

                if self.args.res:
                    output = output + inputs

                L2_loss =  self.loss(output,target) 
                
                if self.args.style_loss > 0:
                    vgg_loss = self.vggloss(output,target,mask)
                else:
                    vgg_loss = 0

                total_loss = L2_loss + self.args.style_loss * vgg_loss

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                losses.update(L2_loss.item(), inputs.size(0))
                
                if self.args.style_loss > 0 :
                    lossvgg.update(vgg_loss.item(), inputs.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss L2: {loss_label:.4f} | Loss VGG: {loss_vgg:.4f}'.format(
                            batch=i + 1,
                            size=len(self.train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_label=losses.avg,
                            loss_vgg=lossvgg.avg
                            )

                if current_index % 1000 == 0:
                    print(suffix)
                
                if self.args.freq > 0 and current_index % self.args.freq == 0:
                    self.validate(current_index)
                    self.flush()
                    self.save_checkpoint()
        
        self.record('train/loss_L2', losses.avg, current_index)


    def test(self, ):

        # switch to evaluate mode
        self.model.eval()

        ssimes = AverageMeter()
        psnres = AverageMeter()

        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask =batches['mask'].to(self.device)

                feeded = torch.cat([inputs,mask],dim=1)
                feeded = feeded.to(self.device)

                output = self.model(feeded)

                if self.args.res:
                    output = output + inputs

                # recover the image to 255
                output = im_to_numpy(torch.clamp(output[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                target = im_to_numpy(torch.clamp(target[0]*255,min=0.0,max=255.0)).astype(np.uint8)

                skimage.io.imsave('%s/%s'%(self.args.checkpoint,batches['name'][0]), output)

                psnr = compare_psnr(target,output)
                ssim = compare_ssim(target,output,multichannel=True)

                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))

        print("%s:PSNR:%s,SSIM:%s"%(self.args.checkpoint,psnres.avg,ssimes.avg))
        print("DONE.\n")
              
        
    def validate(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        ssimes = AverageMeter()
        psnres = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask =batches['mask'].to(self.device)
                
                feeded = torch.cat([inputs,mask],dim=1)
                feeded = feeded.to(self.device)

                output = self.model(feeded)

                if self.args.res:
                    output = output + inputs

                L2_loss = self.loss(output, target)

                psnr = 10 * log10(1 / L2_loss.item())   
                ssim = pytorch_ssim.ssim(output, target)    

                losses.update(L2_loss.item(), inputs.size(0))
                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        print("Epoches:%s,Losses:%.3f,PSNR:%.3f,SSIM:%.3f"%(epoch+1, losses.avg,psnres.avg,ssimes.avg))
        self.record('val/loss_L2', losses.avg, epoch)
        self.record('val/PSNR', psnres.avg, epoch)
        self.record('val/SSIM', ssimes.avg, epoch)
        
        self.metric = psnres.avg
        
    def resume(self,resume_path):
        if isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                current_checkpoint = torch.load(resume_path)
                if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
                    current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module

                if isinstance(current_checkpoint['optimizer'], torch.nn.DataParallel):
                    current_checkpoint['optimizer'] = current_checkpoint['optimizer'].module

                self.args.start_epoch = current_checkpoint['epoch']
                self.metric = current_checkpoint['best_acc']
                self.model.load_state_dict(current_checkpoint['state_dict'])
                # self.optimizer.load_state_dict(current_checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, current_checkpoint['epoch']))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(resume_path))

    def save_checkpoint(self,filename='checkpoint.pth.tar', snapshot=None):
        is_best = True if self.best_acc < self.metric else False

        if is_best:
            self.best_acc = self.metric

        state = {
                    'epoch': self.current_epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_acc': self.best_acc,
                    'optimizer' : self.optimizer.state_dict() if self.optimizer else None,
                }

        filepath = os.path.join(self.args.checkpoint, filename)
        torch.save(state, filepath)

        if snapshot and state['epoch'] % snapshot == 0:
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))
        
        if is_best:
            self.best_acc = self.metric
            print('Saving Best Metric with PSNR:%s'%self.best_acc)
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'model_best.pth.tar'))

    def clean(self):
        self.writer.close()

    def record(self,k,v,epoch):
        self.writer.add_scalar(k, v, epoch)

    def flush(self):
        self.writer.flush()
        sys.stdout.flush()

    def norm(self,x):
        if self.args.gan_norm:
            return x*2.0 - 1.0
        else:
            return x

    def denorm(self,x):
        if self.args.gan_norm:
            return (x+1.0)/2.0
        else:
            return x

