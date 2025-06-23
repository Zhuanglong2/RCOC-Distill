"""
Original implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren
Modified version (CMRNet) by Daniele Cattaneo
Modified version (LCCNet) by Xudong Lv
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
#from pointnet.CMRNet.modules.attention import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import os
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from thop import profile

from models.DyConv import ODConv2d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# from .networks.submodules import *
# from .networks.correlation_package.correlation import Correlation
from models.correlation_package.corre import Correlation


# __all__ = [
#     'calib_net'
# ]

#flownet-Correlation
class Correlation_flownet(nn.Module):
    def __init__(self, max_displacement=40):
        super(Correlation_flownet, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1
        self.pad_size = max_displacement

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        # 输入x2进行padding
        x2_padded = F.pad(x2, (self.pad_size,) * 4)  # 左右上下各填充

        # 使用unfold提取滑动窗口
        x2_unfold = F.unfold(
            x2_padded,
            kernel_size=self.kernel_size,
            stride=1
        ).view(b, c, self.kernel_size ** 2, h, w)

        # 计算逐位置点积
        corr = torch.einsum('bchw,bcshw->bshw', x1, x2_unfold)

        # 归一化处理
        return corr / torch.sqrt(torch.tensor(c, dtype=torch.float))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.elu = nn.ELU()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context


class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRELU = nn.LeakyReLU(0.1)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ECAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer_conv(planes * self.expansion, ratio=reduction)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ModifiedSCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = DPCSAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = PAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = CAlayer(planes * self.expansion, ratio=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class DepthEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self):
        super(DepthEncoder, self).__init__()
        input_channels = 256
        self.up1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(input_channels // 2, input_channels // 4, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(input_channels // 4, input_channels // 8, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(input_channels // 8, input_channels // 16, kernel_size=2, stride=2)

        self.convdepth = nn.Conv2d(input_channels // 16, 1, kernel_size=1)

    def forward(self, input_image):
        feature = self.up1(input_image[4])
        feature = self.up2(feature)
        feature = self.up3(feature)
        feature = self.up4(feature)

        depth = self.convdepth(feature)

        return depth

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, input_channels=1,dyconv = False):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.dyconv = dyconv
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained=False, in_channels=input_channels, dyconv=self.dyconv)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.maxpool(self.features[-1]))
        self.features.append(self.encoder.layer1(self.features[-1]))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class DepthNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self):

        super(DepthNet, self).__init__()
        self.res_num = 18
        # original resnet
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=False, input_channels=3)
        self.depth_decoder = DepthEncoder()

    def forward(self, rgb_img, test):
        if test:
            features = self.net_encoder(rgb_img)
            return features
        else:
            features = self.net_encoder(rgb_img)
            predict_depth_image = self.depth_decoder(features)

            return predict_depth_image, features

class CMAMoudle(nn.Module):

    def __init__(self, ):
        super(CMAMoudle, self).__init__()

        self.convd1 = ODConv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.convd2 = ODConv2d(128, 64, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)

    def forward(self, radarfeature):  # (1, 3, 256, 512)  (1, 1, 256, 512)

        feature = self.convd1(radarfeature[4])
        feature = self.convd2(feature)
        feature = self.up1(feature)
        feature = self.up2(feature) # 16 24

        radarfeature[4] = feature

        return radarfeature

class TestNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):
        super(TestNet, self).__init__()

        self.res_num = res_num      # 18

        # original resnet
        self.pretrained_encoder = True
        self.RGBEncoder = ResnetEncoder(num_layers=self.res_num, pretrained=False, input_channels=3)
        self.net_encoder1 = ResnetEncoder(num_layers=self.res_num, pretrained=False, input_channels=1, dyconv=True)
        self.net_encoder2 = ResnetEncoder(num_layers=self.res_num, pretrained=False, input_channels=1, dyconv=True)

        self.CMA = CMAMoudle()

        self.attention1 = selfAttention(2, 384, 32)
        self.attention2 = selfAttention(2, 384, 32)

        # lidar_image
        # self.corr = Correlation_flownet(max_displacement=20)
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.convmisdepth = nn.Conv2d(512, 81, kernel_size=1)
        self.convlast1 = nn.Conv2d(81, 32, kernel_size=1)#81
        self.convlast2 = nn.Conv2d(81, 32, kernel_size=1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(1024, 512)

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        # mask[mask<0.9999] = 0.0
        # mask[mask>0] = 1.0
        mask = torch.floor(torch.clamp(mask, 0, 1))

        return output * mask

    def forward(self, rgb_img, depth_radar, depth_lidar, test=False):  # (1, 3, 256, 512)  (1, 1, 256, 512)
        #encoder
        features2_distillation = None
        if test is False:
            features2_distillation = self.net_encoder1(depth_lidar)
        #RGB
        features1 = self.RGBEncoder(rgb_img)
        d1 = features1[4]

        # misdepth_image
        features2 = self.net_encoder2(depth_radar)
        features2 = self.CMA(features2)

        md1 = features2[4]  # 32 (1, 512, 8, 16)

        corr = self.corr(d1, md1)  # (1, 256, 16, 24)
        corr = self.leakyRELU(corr) # (1, 81, 16, 24)

        x1 = self.convlast1(corr) #(1, 32, 16, 24)
        x2 = self.convlast2(corr)

        x1 = x1.view(x1.shape[0], x1.shape[1], -1)
        x2 = x2.view(x2.shape[0], x2.shape[1], -1)
        x1 = self.attention1(x1) #(1 32 32)
        x2 = self.attention2(x2)#(1 32 32)

        x1 = x1.view(x1.shape[0], -1)
        x1 = self.dropout(x1)
        x1 = self.leakyRELU(self.fc1(x1))     # (1, 512)
        x2 = x2.view(x2.shape[0], -1)
        x2 = self.dropout(x2)
        x2 = self.leakyRELU(self.fc2(x2))  # (1, 512)

        transl = self.leakyRELU(self.fc1_trasl(x1))# (1, 256)
        rot = self.leakyRELU(self.fc1_rot(x2))# (1, 256)

        transl = self.fc2_trasl(transl) # 1 3
        rot = self.fc2_rot(rot) # 1 4
        rot = F.normalize(rot, dim=1)

        # if test is True:
        #     return transl, rot
        if test is True:
            return transl, rot
        else:
            return transl, rot, features1, features2_distillation


import torch
import time


def calculate_fps(model, inputs, device='cuda', num_warmup=10, num_test=100):
    """
    计算模型推理FPS
    参数：
        model : 待测试模型
        input_size : 输入张量尺寸 (batch, channel, height, width)
        device : 测试设备 cuda/cpu
        num_warmup : 预热次数
        num_test : 正式测试次数
    """
    model.to(device)
    model.eval()


    # 预热阶段
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(inputs[0],inputs[1],inputs[2],inputs[3])

    # CUDA同步计时
    torch.cuda.synchronize()
    start_time = time.time()

    # 正式测试
    with torch.no_grad():
        for _ in range(num_test):
            _ = model(inputs[0],inputs[1],inputs[2],inputs[3])

    # CUDA同步计时
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # 计算指标
    avg_time = elapsed / num_test
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time * 1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
    return fps


from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    rgb_img = torch.randn(1, 3, 256, 384).to('cuda')
    misdepth_img = torch.randn(1, 1, 256, 384).to('cuda')
    LIDAR_img = torch.randn(1, 1, 256, 384).to('cuda')

    feat = 1
    md = 4
    use_reflectance = False
    dropout = 0.0

    model = TestNet((256, 384), use_feat_from=feat, md=md,
                   use_reflectance=use_reflectance, dropout=dropout,
                   Action_Func='leakyrelu', attention=False, res_num=18)
    model = model.cuda()

    transl_err, rot_err,features2, features2_distillation = model(rgb_img, misdepth_img, LIDAR_img,False)

    # print(transl_err.shape)
    # print(rot_err.shape)
    flops, params = profile(model, inputs=(rgb_img, misdepth_img,LIDAR_img,False, ))

    print("FLOPs=", str((flops) / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

    # 测试参数设置
    calculate_fps(model,
                  inputs=(rgb_img, misdepth_img, LIDAR_img,True),  # 根据模型输入调整
                  device='cuda' if torch.cuda.is_available() else 'cpu')

    # flops = FlopCountAnalysis(model, (rgb_img, misdepth_img, LIDAR_img,False)).total()
    # params = parameter_count_table(model)
    # print(f"FLOPs: {flops / 1e9:.2f}G, Params:\n{params}")