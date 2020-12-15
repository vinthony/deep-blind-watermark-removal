

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from scripts.utils.model_init import *
from scripts.models.vgg import Vgg16
from scripts.models.blocks import *


class CAWapper(nn.Module):
    """docstring for SENet"""

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(CAWapper, self).__init__()
        self.attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=True, use_cuda=True)

    def forward(self, feature, mask):
        _, _, w, _ = feature.size()
        _, _, mw, _ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        mask = torch.round(F.avg_pool2d(mask, 2, stride=mw//w))

        result = self.attention(feature,mask)

        return result


class NLWapper(nn.Module):
    """docstring for SENet"""

    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(NLWapper, self).__init__()
        self.attention = NONLocalBlock2D(channel)

    def forward(self, feature, mask):
        _, _, w, _ = feature.size()
        _, _, mw, _ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        # mask = torch.round(F.avg_pool2d(mask, 2, stride=mw//w))

        result = self.attention(feature)

        return result

class SENet(nn.Module):
    """docstring for SENet"""
    def __init__(self,channel,type_of_connection=BasicLearningBlock):
        super(SENet, self).__init__()
        self.attention = SEBlock(channel,16)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))

        result = self.attention(feature) 
        
        return result

class CBAMConnect(nn.Module):
    def __init__(self,channel):
        super(CBAMConnect, self).__init__()
        self.attention = CBAM(channel)

    def forward(self,feature,mask):
        results = self.attention(feature)
        return results



class RASC(nn.Module):
    def __init__(self,channel,type_of_connection=BasicLearningBlock):
        super(RASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.gaussianMask = GaussianSmoothing(1,5,1)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        if w != mw:
            mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        # here we add gaussin filter to mask and reverse_mask for better harimoization of edges.

        mask = self.gaussianMask(F.pad(mask,(2,2,2,2),mode='reflect'))
        reverse_mask = self.gaussianMask(F.pad(reverse_mask,(2,2,2,2),mode='reflect'))


        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced    


class UNO(nn.Module):
    def __init__(self,channel):
        super(UNO, self).__init__()

    def forward(self,feature,_m):
        return feature 


class URASC(nn.Module):
    def __init__(self,channel,type_of_connection=BasicLearningBlock):
        super(URASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.mask_attention = SpatialAttentionModule(channel,16)

    def forward(self,feature, m=None):
        _,_,w,_ = feature.size()
      
        mask, reverse_mask = self.mask_attention(feature)

        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced  


class MaskedURASC(nn.Module):
    def __init__(self,channel,type_of_connection=BasicLearningBlock):
        super(MaskedURASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.mask_attention = SpatialAttentionModule(channel,16)

    def forward(self,feature):
        _,_,w,_ = feature.size()
      
        mask, reverse_mask = self.mask_attention(feature)

        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature) 
        spliced = ( self.connection(spliced_feature) + selected_feature ) * mask
        return background + spliced, mask

