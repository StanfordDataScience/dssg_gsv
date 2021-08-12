"""models.py
-------------------------------------------------------------------------
Created by: Shubhang Desai
Date created: October 2020
Last revised: June 15, 2021
Project: GSV
Subproject: ml-buildings
-------------------------------------------------------------------------

Returns a model to be consumed by the training or evaluation script.
"""


import torch
import torch.nn as nn
import torchvision.models as models

import os

from new_models.lenet import *


class DecoderBlock(nn.Module):
    """
    Decoder block for UNet model
    """

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle'):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode=='deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels = out_channels,
                                                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode=='pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)

        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x=self.conv(x)
        x=self.norm1(x)
        x=self.relu1(x)
        x=self.upsample(x)
        x=self.norm2(x)
        x=self.relu2(x)
        return x

class UNet(nn.Module):
    """
    UNet model
    """

    def __init__(self, num_classes=40):
        super(UNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        filters=[64, 256, 512, 1024]

        self.conv = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3])
        self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2])
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1])
        self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0])

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_ = self.maxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center,e2],dim=1))
        d3 = self.decoder2(torch.cat([d2,e1], dim=1))
        d4 = self.decoder3(torch.cat([d3,x], dim=1))
        out = self.final(d4)

        return out

class ResNet18_features(nn.Module):
    def __init__(self, N):
        super(ResNet18_features, self).__init__()
        model = models.resnet18(pretrained=True)
        
        self.out_dim = {
            1: 64,
            2: 128,
            3: 256
        }[N]

        self.features = nn.Sequential(
            *list(model.children())[:N+4]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(self.out_dim, 1000)

    def forward(self, x):
        x = self.features(x)
        x = torch.squeeze(self.avgpool(x))
        x = self.fc(x)
        return x

def transform(scores, bias, gain):
    return gain * (scores - 25) + bias + 25

class Ensemble(nn.Module):
    """
    Ensemble that computes a composite score from multiple constituent models by scaling and averaging.
    """
    def __init__(self, models, bias, gain):
        self.models = nn.ModuleList(models)
        self.bias = bias
        self.gain = gain
    
    def forward(self, x):
        n = len(self.models)
        scores = [None] * n
        for i in range(n):
            scores[i] = transform(self.models[i](x), bias[i], gain[i])
        all_scores = torch.stack(scores, dim=0)
        score = torch.mean(all_scores, dim=0, keepdim=False)
        return score

def get_model(args):
    """
    Creates model to be used for training

    Paramters
    ---------
    args : dict
        dictionary of arguments from get_args()

    Returns
    -------
    model : torch.nn.Module
        model to be used for training

    """

    if args['pretrain']:
        return UNet()
      
    
    model = models.resnet18(pretrained=True)
    """
    model = models.resnet18(pretrained=False)
    N = 3
    model = ResNet18_features(N)
    """

    out_features = {
        'regr': 1,
        'mc': 3,
        'pa': 2,
        'lh': 2
    }[args['label_type']]
 
    """
    #for param in model.parameters():
    #    param.requires_grad = False
 
    #model = LeNet5(out_features)
    """
    model.fc = nn.Linear(model.fc.in_features, out_features)


    if args['pretrain_path'] != '':
        ckpts = os.path.join(args['pretrain_path'], 'checkpoints')
        sorted_ckpts = sorted((int(p.split('_')[0]), p) for p in os.listdir(ckpts))
        best_ckpt = os.path.join(ckpts, sorted_ckpts[-1][1])

        seg_model = UNet()
        seg_model.load_state_dict(torch.load(best_ckpt))

        model.conv1 = seg_model.conv
        model.bn = seg_model.bn
        model.relu = seg_model.relu
        model.maxpool = seg_model.maxpool
        model.layer1 = seg_model.encoder1
        model.layer2 = seg_model.encoder2
        model.layer3 = seg_model.encoder3

    return model
