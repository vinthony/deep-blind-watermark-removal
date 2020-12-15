import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.models.vgg import Vgg19
from torchvision import models
from scripts.utils.misc import resize_to_match
# from pytorch_msssim import SSIM, MS_SSIM
import pytorch_ssim

class WeightedBCE(nn.Module):
    def __init__(self):
        super(WeightedBCE, self).__init__()

    def forward(self, pred, gt):
        eposion = 1e-10
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)

        return loss


def l1_relative(reconstructed, real, mask):
    batch = real.size(0)
    area = torch.sum(mask.view(batch,-1),dim=1)
    reconstructed = reconstructed * mask
    real = real * mask
    
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1


def is_dic(x):
    return type(x) == type([])

class Losses(nn.Module):
    def __init__(self, argx, device):
        super(Losses, self).__init__()
        self.args = argx

        if self.args.loss_type == 'l1bl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.L1Loss(), nn.BCELoss(), nn.MSELoss()
        elif self.args.loss_type == 'l1wbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.L1Loss(), WeightedBCE(), nn.MSELoss() 
        elif self.args.loss_type == 'l2wbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), WeightedBCE(), nn.MSELoss()
        elif self.args.loss_type == 'l2xbl2':
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.MSELoss()
        else: # l2bl2
            self.outputLoss, self.attLoss, self.wrloss = nn.MSELoss(), nn.BCELoss(), nn.MSELoss()

        if self.args.style_loss > 0:
            self.vggloss = VGGLoss(self.args.sltype).to(device)
        
        if self.args.ssim_loss > 0:
            self.ssimloss =  pytorch_ssim.SSIM().to(device)

        self.outputLoss = self.outputLoss.to(device)
        self.attLoss = self.attLoss.to(device)
        self.wrloss = self.wrloss.to(device)


    def forward(self,imgx,target,attx,mask,wmx,wm):
        pixel_loss,att_loss,wm_loss,vgg_loss,ssim_loss = 0,0,0,0,0

        if is_dic(imgx):

            if self.args.masked:
            # calculate the overall loss and side output
                pixel_loss = self.outputLoss(imgx[0],target) + sum([self.outputLoss(im,resize_to_match(mask,im)*resize_to_match(target,im)) for im in imgx[1:]])
            else:
                pixel_loss =  sum([self.outputLoss(im,resize_to_match(target,im)) for im in imgx])

            if self.args.style_loss > 0:
                vgg_loss = sum([self.vggloss(im,resize_to_match(target,im),resize_to_match(mask,im)) for im in imgx])

            if self.args.ssim_loss > 0:
                ssim_loss = sum([ 1 - self.ssimloss(im,resize_to_match(target,im)) for im in imgx])
        else:

            if self.args.masked:
                pixel_loss = self.outputLoss(imgx,mask*target)
            else:
                pixel_loss =  self.outputLoss(imgx,target)

            if self.args.style_loss > 0:
                vgg_loss = self.vggloss(imgx,target,mask)

            if self.args.ssim_loss > 0:
                ssim_loss = 1 - self.ssimloss(imgx,target)

        if is_dic(attx):
            att_loss =  sum([self.attLoss(at,resize_to_match(mask,at)) for at in attx])
        else:
            att_loss =  self.attLoss(attx, mask)

        if is_dic(wmx):
            wm_loss = sum([self.wrloss(w,resize_to_match(wm,w)) for w in wmx])
        else:
            if self.args.masked:
                wm_loss = self.wrloss(wmx,mask*wm)
            else:
                wm_loss = self.wrloss(wmx, wm)

        return pixel_loss,att_loss,wm_loss,vgg_loss,ssim_loss



def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram
    
class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False



def VGGLoss(losstype):
    if losstype == 'vgg':
        return VGGLossA()
    elif losstype == 'vggx':
        return VGGLossX(mask=False)
    elif losstype == 'mvggx':
        return VGGLossX(mask=True)
    elif losstype == 'rvggx':
        return VGGLossX(mask=True,relative=True)
    else:
        raise Exception("error in %s"%losstype)

        

class VGGLossA(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLossA, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class VGGLossX(nn.Module):
    def __init__(self, normalize=True, mask=False, relative=False):
        super(VGGLossX, self).__init__()
        
        self.vgg = VGG16FeatureExtractor().cuda()
        self.criterion = nn.L1Loss().cuda() if not relative else l1_relative
        self.use_mask= mask
        self.relative = relative

        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y, Xmask=None):
        if not self.use_mask:
            mask = torch.ones_like(x)[:,0:1,:,:]
        else:
            mask = Xmask

        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)

        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        
        loss = 0
        for i in range(3):
            if self.relative:
                loss += self.criterion(x_vgg[i],y_vgg[i].detach(),resize_to_match(mask,x_vgg[i]))
            else:
                loss += self.criterion(resize_to_match(mask,x_vgg[i])*x_vgg[i],resize_to_match(mask,y_vgg[i])*y_vgg[i].detach())

        return loss


class GANLosses(object):
    """docstring for Loss"""
    def __init__(self, gantype):
        super(GANLosses, self).__init__()        
        self.generator_loss = gen_gan(gantype)
        self.discriminator_loss = dis_gan(gantype)
        self.gantype = gantype

    def g_loss(self,dis_fake):
        if 'hinge' in self.gantype:
            return gen_hinge(dis_fake)
        else:
            return self.generator_loss(dis_fake)

    def d_loss(self,dis_fake,dis_real):
        if 'hinge' in self.gantype:
            return dis_hinge(dis_fake,dis_real)
        else:
            return self.discriminator_loss(dis_fake,dis_real)


class gen_gan(nn.Module):
    def __init__(self,gantype):
        super(gen_gan,self).__init__()
        if gantype == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gantype == 'naive':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise Exception("error gan type")
    
    def forward(self,dis_fake):
        return self.criterion(dis_fake, torch.ones_like(dis_fake))

class dis_gan(nn.Module):
    def __init__(self,gantype):
        super(dis_gan,self).__init__()
        if gantype == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gantype == 'naive':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise Exception("error gan type")
    
    def forward(self,dis_fake,dis_real):
        loss_fake = self.criterion(dis_fake, torch.zeros_like(dis_fake))
        loss_real = self.criterion(dis_real, torch.ones_like(dis_real))
        return loss_fake, loss_real

# def gen_gan(dis_fake):
#     # fake -> 1
#     return F.binary_cross_entropy_with_logits(dis_fake,torch.ones_like(dis_fake))

# def dis_gan(dis_fake,dis_real):
#     # fake -> 0 , real ->1
#     loss_fake = F.binary_cross_entropy_with_logits(dis_fake, torch.zeros_like(dis_real))
#     loss_real = F.binary_cross_entropy_with_logits(dis_real, torch.ones_like(dis_fake))
#     return loss_fake,loss_real 

# def gen_lsgan(dis_fake):
#     loss = F.mse_loss(dis_fake,torch.ones_like(dis_fake)) # 
#     return loss

# def dis_lsgan(dis_fake, dis_real):
#     loss_fake = F.mse_loss(dis_fake, torch.zeros_like(dis_real))
#     loss_real = F.mse_loss(dis_real, torch.ones_like(dis_real))
#     return loss_fake,loss_real

def gen_hinge(dis_fake, dis_real=None):
    return -torch.mean(dis_fake)

def dis_hinge(dis_fake, dis_real):
    loss_fake = torch.mean(torch.relu(1. + dis_fake))
    loss_real = torch.mean(torch.relu(1. - dis_real))
    return loss_fake,loss_real

