
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.models.blocks import SEBlock
from scripts.models.rasc import *
from scripts.models.unet import UnetGenerator,MinimalUnetV2

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def up_conv2x2(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)




class UpCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, residual=True, batch_norm=True, transpose=True,concat=True,use_att=False):
        super(UpCoXvD, self).__init__()
        self.concat = concat
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv2 = []
        self.use_att = use_att
        self.up_conv = up_conv2x2(in_channels, out_channels, transpose=transpose)
        
        if self.use_att:
            self.s2am = RASC(2 * out_channels)
        else:
            self.s2am = None

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)

    def forward(self, from_up, from_down, mask=None):
        from_up = self.up_conv(from_up)
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up

        if self.use_att:
            x1 = self.s2am(x1,mask)

        x1 = F.relu(self.conv1(x1))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        return x2


class DownCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, residual=True, batch_norm=True):
        super(DownCoXvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool

class UnetDecoderD(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True):
        super(UnetDecoderD, self).__init__()
        self.conv_final = None
        self.up_convs = []
        outs = in_channels
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat)
            self.up_convs.append(up_conv)
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    def forward(self, x, encoder_outs=None):
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            x = up_conv(x, before_pool)
        if self.conv_final is not None:
            x = self.conv_final(x)
        return x


class UnetEncoderD(nn.Module):

    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True):
        super(UnetEncoderD, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownCoXvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs



class UnetVM(nn.Module):

    def __init__(self, in_channels=3, depth=5, shared_depth=0, use_vm_decoder=False, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True, batch_norm=True,
                 transpose=True, concat=True, transfer_data=True, long_skip=False):
        super(UnetVM, self).__init__()
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder,  self.optimizer_image, self.optimizer_vm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks)
        if not transfer_data:
            concat = False
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=depth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, batch_norm=batch_norm)
        self.image_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                          out_channels=out_channels_image, depth=depth - shared_depth,
                                          blocks=blocks[1], residual=residual, batch_norm=batch_norm,
                                          transpose=transpose, concat=concat)
        self.mask_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1),
                                         out_channels=out_channels_mask, depth=depth,
                                         blocks=blocks[2], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat)
        self.vm_decoder = None
        if use_vm_decoder:
            self.vm_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                           out_channels=out_channels_image, depth=depth - shared_depth,
                                           blocks=blocks[3], residual=residual, batch_norm=batch_norm,
                                           transpose=transpose, concat=concat)
        self.shared_decoder = None
        self.long_skip = long_skip
        self._forward = self.unshared_forward
        if self.shared != 0:
            self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderD(in_channels=start_filters * 2 ** (depth - 1),
                                               out_channels=start_filters * 2 ** (depth - shared_depth - 1),
                                               depth=shared_depth, blocks=blocks[4], residual=residual,
                                               batch_norm=batch_norm, transpose=transpose, concat=concat,
                                               is_final=False)

    def set_optimizers(self):
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optimizer_image = torch.optim.Adam(self.image_decoder.parameters(), lr=0.001)
        self.optimizer_mask = torch.optim.Adam(self.mask_decoder.parameters(), lr=0.001)
        if self.vm_decoder is not None:
            self.optimizer_vm = torch.optim.Adam(self.vm_decoder.parameters(), lr=0.001)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=0.001)

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        self.optimizer_mask.zero_grad()
        if self.vm_decoder is not None:
            self.optimizer_vm.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()
        self.optimizer_image.step()
        self.optimizer_mask.step()
        if self.vm_decoder is not None:
            self.optimizer_vm.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    def step_optimizer_image(self):
        self.optimizer_image.step()

    def __call__(self, synthesized):
        return self._forward(synthesized)

    def forward(self, synthesized):
        return self._forward(synthesized)

    def unshared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if not self.transfer_data:
            before_pool = None
        reconstructed_image = torch.tanh(self.image_decoder(image_code, before_pool))
        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(image_code, before_pool))
            return reconstructed_image, reconstructed_mask, reconstructed_vm
        return reconstructed_image, reconstructed_mask

    def shared_forward(self, synthesized):
        image_code, before_pool = self.encoder(synthesized)
        if self.transfer_data:
            shared_before_pool = before_pool[- self.shared - 1:]
            unshared_before_pool = before_pool[: - self.shared]
        else:
            before_pool = None
            shared_before_pool = None
            unshared_before_pool = None
        x = self.shared_decoder(image_code, shared_before_pool)
        reconstructed_image = torch.tanh(self.image_decoder(x, unshared_before_pool))
        if self.long_skip:
            reconstructed_image = reconstructed_image + synthesized

        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        if self.vm_decoder is not None:
            reconstructed_vm = torch.tanh(self.vm_decoder(x, unshared_before_pool))
            if self.long_skip:
                reconstructed_vm = reconstructed_vm + synthesized
            return reconstructed_image, reconstructed_mask, reconstructed_vm
        return reconstructed_image, reconstructed_mask
