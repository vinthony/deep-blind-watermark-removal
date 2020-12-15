import torch
import torch.nn as nn
from torch.nn import init
import functools
from scripts.models.blocks import *
from scripts.models.rasc import *


class MinimalUnetV2(nn.Module):
    """docstring for MinimalUnet"""
    def __init__(self, down=None,up=None,submodule=None,attention=None,withoutskip=False,**kwags):
        super(MinimalUnetV2, self).__init__()
        
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up) 
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None 
        self.is_sub = not submodule == None 
    
    def forward(self,x,mask=None):
        if self.is_sub: 
            x_up,_ = self.sub(self.down(x),mask)
        else:
            x_up = self.down(x)

        if self.withoutskip: #outer or inner.
            x_out = self.up(x_up)
        else:
            if self.is_attention:
                x_out = (self.attention(torch.cat([x,self.up(x_up)],1),mask),mask)
            else:
                x_out = (torch.cat([x,self.up(x_up)],1),mask)

        return x_out


class MinimalUnet(nn.Module):
    """docstring for MinimalUnet"""
    def __init__(self, down=None,up=None,submodule=None,attention=None,withoutskip=False,**kwags):
        super(MinimalUnet, self).__init__()
        
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up) 
        self.sub = submodule
        self.attention = attention
        self.withoutskip = withoutskip
        self.is_attention = not self.attention == None 
        self.is_sub = not submodule == None 
    
    def forward(self,x,mask=None):
        if self.is_sub: 
            x_up,_ = self.sub(self.down(x),mask)
        else:
            x_up = self.down(x)

        if self.is_attention:
            x = self.attention(x,mask)
        
        if self.withoutskip: #outer or inner.
            x_out = self.up(x_up)
        else:
            x_out = (torch.cat([x,self.up(x_up)],1),mask)

        return x_out


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,is_attention_layer=False,
                 attention_model=RASC,basicblock=MinimalUnet,outermostattention=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)


        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = basicblock(down,up,submodule,withoutskip=outermost)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = basicblock(down,up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if is_attention_layer:
                if MinimalUnetV2.__qualname__ in basicblock.__qualname__  :
                    attention_model = attention_model(input_nc*2)
                else:
                    attention_model = attention_model(input_nc)     
            else:
                attention_model = None
                
            if use_dropout:
                model = basicblock(down,up.append(nn.Dropout(0.5)),submodule,attention_model,outermostattention=outermostattention)
            else:
                model = basicblock(down,up,submodule,attention_model,outermostattention=outermostattention)

        self.model = model


    def forward(self, x,mask=None):
        # build the mask for attention use
        return self.model(x,mask)
            
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False,
                 is_attention_layer=False,attention_model=RASC,use_inner_attention=False,basicblock=MinimalUnet):
        super(UnetGenerator, self).__init__()

        # 8 for 256x256
        # 9 for 512x512
        # construct unet structure
        self.need_mask = not input_nc == output_nc

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,basicblock=basicblock) # 1
        for i in range(num_downs - 5): #3 times
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,is_attention_layer=use_inner_attention,attention_model=attention_model,basicblock=basicblock) # 8,4,2
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock) #16
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock) #32
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,is_attention_layer=is_attention_layer,attention_model=attention_model,basicblock=basicblock, outermostattention=True) #64 
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, basicblock=basicblock, norm_layer=norm_layer) # 128

        self.model = unet_block

    def forward(self, input):
        if self.need_mask:
            return self.model(input,input[:,3:4,:,:])
        else:
            return self.model(input[:,0:3,:,:],input[:,3:4,:,:])



