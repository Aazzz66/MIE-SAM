import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union, List
import numpy as np
import timm
import cv2
import torch.autograd as autograd
from models.sam import sam_model_registry
import cfg as cfg
import matplotlib.pyplot as plt
from models.DFM import DynamicFusionModule

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )



class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )



class Net(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=2,
                 ):
        super().__init__()
        args = cfg.parse_args()
        # self.sam = sam_model_registry["vit_b"](args,checkpoint='weights/sam_vit_b_01ec64.pth')
        self.sam = sam_model_registry["vit_l"](args, checkpoint='weights/sam_vit_l_0b3195.pth')
        # self.sam = sam_model_registry["vit_h"](args,checkpoint='weights/sam_vit_h_4b8939.pth')
        self.image_encoder = self.sam.image_encoder
        encoder_channels = (256, 256, 256, 256)
        self.fusion_module = DynamicFusionModule(
            in_channels_per_feat=256,
            filter_num=1,
            feat_num=2,
            intermediate_channels=[256, 128, 64, 32],
            filter_type='conv2d'  # 或 'conv2d'
        )
        self.PDM_upsample = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.PDM_head = nn.Sequential(ConvBNReLU(256, 128),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               ConvBNReLU(128, 32),
                                               Conv(32, num_classes, kernel_size=1))
        self.softmax = nn.Softmax(dim=1)
        self.init_weight()

        for n, value in self.image_encoder.named_parameters():
            if 'lora_' not in n:
                value.requires_grad = False
        else:
            value.requires_grad = True

        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False

        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = False

        # self.decoder = Decoder_single(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x, y,mode=None):
        h, w = x.size()[-2:]
        deepx, deepy = self.image_encoder(x, y)
        fuse_feats, fuse_feat_masks = self.fusion_module(feat_list=[deepx, deepy])
        # print('Fused feature shape:', fuse_feats.shape, fuse_feat_masks.shape)
        fuse_feats1 = self.PDM_upsample(fuse_feats)
        x = self.PDM_head(fuse_feats1)
        x = self.softmax(x)[:, 1:2, :, :]
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)