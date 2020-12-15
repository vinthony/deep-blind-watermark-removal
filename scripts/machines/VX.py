import torch
import torch.nn as nn
from progress.bar import Bar
from tqdm import tqdm
import pytorch_ssim
import json
import sys,time,os
import torchvision
from math import log10
import numpy as np
from .BasicMachine import BasicMachine
from scripts.utils.evaluation import accuracy, AverageMeter, final_preds
from scripts.utils.misc import resize_to_match
from torch.autograd import Variable
import torch.nn.functional as F
from scripts.utils.parallel import DataParallelModel, DataParallelCriterion
from scripts.utils.losses import VGGLoss, l1_relative,is_dic
from scripts.utils.imutils import im_to_numpy
import skimage.io
from skimage.measure import compare_psnr,compare_ssim


class Losses(nn.Module):
    def __init__(self, argx, device, norm_func=None, denorm_func=None):
        super(Losses, self).__init__()
        self.args = argx

        if self.args.loss_type == 'l1bl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.L1Loss(), nn.BCELoss(), nn.MSELoss()
        elif self.args.loss_type == 'l2xbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.MSELoss()
        elif self.args.loss_type == 'relative' or self.args.loss_type == 'hybrid':
            self.outputLoss, self.attLoss, self.wrloss = l1_relative, nn.BCELoss(), l1_relative
        else: # l2bl2
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCELoss(), nn.MSELoss()

        self.default = nn.L1Loss()

        if self.args.style_loss > 0:
            self.vggloss = VGGLoss(self.args.sltype).to(device)
        
        if self.args.ssim_loss > 0:
            self.ssimloss =  pytorch_ssim.SSIM().to(device)
        
        self.norm = norm_func
        self.denorm = denorm_func


    def forward(self,pred_ims,target,pred_ms,mask,pred_wms,wm):
        pixel_loss,att_loss,wm_loss,vgg_loss,ssim_loss = [0]*5
        pred_ims = pred_ims if is_dic(pred_ims) else [pred_ims]

        # try the loss in the masked region
        if self.args.masked and 'hybrid' in self.args.loss_type: # masked loss
            pixel_loss += sum([self.outputLoss(pred_im, target, mask) for pred_im in pred_ims])
            pixel_loss += sum([self.default(pred_im*pred_ms,target*mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*mask + (1-mask)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss += self.wrloss(pred_wms, wm, mask)
            wm_loss += self.default(pred_wms*pred_ms, wm*mask)

        elif self.args.masked and 'relative' in self.args.loss_type: # masked loss
            pixel_loss += sum([self.outputLoss(pred_im, target, mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*mask + (1-mask)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss = self.wrloss(pred_wms, wm, mask)
        elif self.args.masked:
            pixel_loss += sum([self.outputLoss(pred_im*mask, target*mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*pred_ms + (1-pred_ms)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss = self.wrloss(pred_wms*mask, wm*mask)
        else:
            pixel_loss += sum([self.outputLoss(pred_im*pred_ms, target*mask) for pred_im in pred_ims])
            recov_imgs = [ self.denorm(pred_im*pred_ms + (1-pred_ms)*self.norm(target)) for pred_im in pred_ims ]
            wm_loss = self.wrloss(pred_wms*pred_ms,wm*mask)

        pixel_loss += sum([self.default(im,target) for im in recov_imgs])

        if self.args.style_loss > 0:
            vgg_loss = sum([self.vggloss(im,target,mask) for im in recov_imgs])

        if self.args.ssim_loss > 0:
            ssim_loss = sum([ 1 - self.ssimloss(im,target) for im in recov_imgs])

        att_loss =  self.attLoss(pred_ms, mask)

        return pixel_loss,att_loss,wm_loss,vgg_loss,ssim_loss


class VX(BasicMachine):
    def __init__(self,**kwargs):
        BasicMachine.__init__(self,**kwargs)
        self.loss = Losses(self.args, self.device, self.norm, self.denorm)
        self.model.set_optimizers()
        self.optimizer = None
       
    def train(self,epoch):

        self.current_epoch = epoch

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossMask = AverageMeter()
        lossWM = AverageMeter()
        lossMX = AverageMeter()
        lossvgg = AverageMeter()
        lossssim = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.train_loader))

        for i, batches in enumerate(self.train_loader):

            current_index = len(self.train_loader) * epoch + i

            inputs = batches['image'].to(self.device)
            target = batches['target'].to(self.device)
            mask = batches['mask'].to(self.device)
            wm =  batches['wm'].to(self.device)

            outputs = self.model(self.norm(inputs))
            
            self.model.zero_grad_all()

            l2_loss,att_loss,wm_loss,style_loss,ssim_loss = self.loss(outputs[0],self.norm(target),outputs[1],mask,outputs[2],self.norm(wm))
            total_loss = 2*l2_loss + self.args.att_loss * att_loss + wm_loss + self.args.style_loss * style_loss + self.args.ssim_loss * ssim_loss

            # compute gradient and do SGD step
            total_loss.backward()
            self.model.step_all()

            # measure accuracy and record loss
            losses.update(l2_loss.item(), inputs.size(0))
            lossMask.update(att_loss.item(), inputs.size(0))
            lossWM.update(wm_loss.item(), inputs.size(0))

            if self.args.style_loss > 0 :
                lossvgg.update(style_loss.item(), inputs.size(0))

            if self.args.ssim_loss > 0 :
                lossssim.update(ssim_loss.item(), inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix  = "({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss L2: {loss_label:.4f} | Loss Mask: {loss_mask:.4f} | loss WM: {loss_wm:.4f} | loss VGG: {loss_vgg:.4f} | loss SSIM: {loss_ssim:.4f}| loss MX: {loss_mx:.4f}".format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_label=losses.avg,
                        loss_mask=lossMask.avg,
                        loss_wm=lossWM.avg,
                        loss_vgg=lossvgg.avg,
                        loss_ssim=lossssim.avg,
                        loss_mx=lossMX.avg
                        )
            if current_index % 1000 == 0:
                print(suffix)

            if self.args.freq > 0 and current_index % self.args.freq == 0:
                self.validate(current_index)
                self.flush()
                self.save_checkpoint()

        self.record('train/loss_L2', losses.avg, epoch)
        self.record('train/loss_Mask', lossMask.avg, epoch)
        self.record('train/loss_WM', lossWM.avg, epoch)
        self.record('train/loss_VGG', lossvgg.avg, epoch)
        self.record('train/loss_SSIM', lossssim.avg, epoch)
        self.record('train/loss_MX', lossMX.avg, epoch)




    def validate(self, epoch):

        self.current_epoch = epoch
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossMask = AverageMeter()
        psnres = AverageMeter()
        ssimes = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing {} '.format(self.args.arch), max=len(self.val_loader))
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):

                current_index = len(self.val_loader) * epoch + i

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)

                outputs = self.model(self.norm(inputs))
                imoutput,immask,imwatermark = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput

                imfinal = self.denorm(imoutput*immask + self.norm(inputs)*(1-immask))

                if i % 300 == 0:
                    # save the sample images
                    ims = torch.cat([inputs,target,imfinal,immask.repeat(1,3,1,1)],dim=3)
                    torchvision.utils.save_image(ims,os.path.join(self.args.checkpoint,'%s_%s.jpg'%(i,epoch)))

                # here two choice: mseLoss or NLLLoss
                psnr = 10 * log10(1 / F.mse_loss(imfinal,target).item())       

                ssim = pytorch_ssim.ssim(imfinal,target)

                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_L2: {loss_label:.4f} | Loss_Mask: {loss_mask:.4f} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f}'.format(
                            batch=i + 1,
                            size=len(self.val_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_label=losses.avg,
                            loss_mask=lossMask.avg,
                            psnr=psnres.avg,
                            ssim=ssimes.avg
                            )
                bar.next()
        bar.finish()
        
        print("Iter:%s,Losses:%s,PSNR:%.4f,SSIM:%.4f"%(epoch, losses.avg,psnres.avg,ssimes.avg))
        self.record('val/loss_L2', losses.avg, epoch)
        self.record('val/lossMask', lossMask.avg, epoch)
        self.record('val/PSNR', psnres.avg, epoch)
        self.record('val/SSIM', ssimes.avg, epoch)
        self.metric = psnres.avg

        self.model.train()

    def test(self, ):

        # switch to evaluate mode
        self.model.eval()
        print("==> testing VM model ")
        ssimes = AverageMeter()
        psnres = AverageMeter()
        ssimesx = AverageMeter()
        psnresx = AverageMeter()

        with torch.no_grad():
            for i, batches in enumerate(tqdm(self.val_loader)):

                inputs = batches['image'].to(self.device)
                target = batches['target'].to(self.device)
                mask =batches['mask'].to(self.device)

                # select the outputs by the giving arch
                outputs = self.model(self.norm(inputs))
                imoutput,immask,imwatermark = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput

                imfinal = self.denorm(imoutput*immask + self.norm(inputs)*(1-immask))
                psnrx = 10 * log10(1 / F.mse_loss(imfinal,target).item())       
                ssimx = pytorch_ssim.ssim(imfinal,target)
                # recover the image to 255
                imfinal = im_to_numpy(torch.clamp(imfinal[0]*255,min=0.0,max=255.0)).astype(np.uint8)
                target = im_to_numpy(torch.clamp(target[0]*255,min=0.0,max=255.0)).astype(np.uint8)

                skimage.io.imsave('%s/%s'%(self.args.checkpoint,batches['name'][0]), imfinal)

                psnr = compare_psnr(target,imfinal)
                ssim = compare_ssim(target,imfinal,multichannel=True)

                psnres.update(psnr, inputs.size(0))
                ssimes.update(ssim, inputs.size(0))
                psnresx.update(psnrx, inputs.size(0))
                ssimesx.update(ssimx, inputs.size(0))

        print("%s:PSNR:%.5f(%.5f),SSIM:%.5f(%.5f)"%(self.args.checkpoint,psnres.avg,psnresx.avg,ssimes.avg,ssimesx.avg))
        print("DONE.\n")