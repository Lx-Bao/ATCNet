import torch
import torch.nn as nn
from torchvision.models import resnet50 as resnet
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .swin_transformer import SwinTransformer
from .Module import SELayer,CBAM,Attention_block,DoubleConv,Conv,ChannelPool




class FF_module(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.,BiSA = True):
        super(FF_module, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.BiSA = BiSA

        self.CBAM_s = CBAM(ch_2)
        self.CBAM_c = CBAM(ch_1)

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_2, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_1, 1, bn=True, relu=False)
        self.bn1 = nn.BatchNorm2d(ch_2)
        self.bn2 = nn.BatchNorm2d(ch_1)

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(in_channels=ch_1 + ch_2, out_channels=ch_out, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.fuse_model = SELayer(ch_1 + ch_2)


        self.relu = nn.ReLU(inplace=True)


        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x,p):
        # bilinear pooling
        W_g = self.W_g(g)
        mul1 = x+self.relu(self.bn1(W_g*x))
        W_x = self.W_x(x)
        mul2 = g+self.relu(self.bn2(W_x*g))
        fuse = torch.cat((mul1,mul2),1)
        fused = self.fuse_model(fuse)

        if self.BiSA:
            fused = fused.mul(p)


        if self.drop_rate > 0:
            return self.dropout(self.conv_fuse(fused+fuse))
        else:
            return self.conv_fuse(fused+fuse)



class RO_module(nn.Module):
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2,S_pre):

        x1 = self.up(x1)
        # input is CHW

        if self.attn_block is not None:
            x2 = self.attn_block(x1, x2,S_pre)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)





class ATC(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(ATC, self).__init__()

        self.resnet = resnet()
        self.swin1 = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
            self.swin1.load_state_dict(torch.load('pretrained/swin_base_patch4_window12_384_22k.pth')['model'],strict=False)
        self.resnet.fc = nn.Identity()


        self.final_4 = nn.Sequential(
            Conv(1024, 256, 1, bn=True, relu=True),
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False),

        )
        
        self.final_3 = nn.Sequential(
            Conv(512, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False),

            )

        self.final_2 = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False),

            )

        self.final_1 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False),

            )

        self.final_0 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False),

        )

        self.FF_5 = FF_module(ch_1=2048, ch_2=1024, r_2=4, ch_int=512, ch_out=1024, drop_rate=drop_rate/2,BiSA=False)
        self.FF_4 = FF_module(ch_1=1024, ch_2=512, r_2=4, ch_int=256, ch_out=512, drop_rate=drop_rate / 2)
        self.FF_3 = FF_module(ch_1=512, ch_2=256, r_2=2, ch_int=128, ch_out=256, drop_rate=drop_rate/2)
        self.FF_2 = FF_module(ch_1=256, ch_2=128,  r_2=1, ch_int=64, ch_out=128, drop_rate=drop_rate/2)
        #self.up_c0 = BiFusion_block_low(ch_1=64, ch_2=128, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate / 2)

        self.RO = RO_module(128, 64, 64, attn=True)

        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout2d(drop_rate)


        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):

        score_list,score_PE = self.swin1(imgs)
        #score_PE = self.up3(score_PE)
        score_T_1 = score_list[0]
        #print(score_T_1.size())
        score_T_2 = score_list[1]
        #print(score_T_2.size())
        score_T_3 = score_list[2]
        #print(score_T_3.size())
        score_T_4 = score_list[3]
        #print(score_T_4.size())

        score_T_4 = self.drop(score_T_4)
        score_T_3 = self.drop(score_T_3)

        score_T_2 = self.drop(score_T_2)

        score_T_1 = self.drop(score_T_1)


        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u0 = x_u
        x_u = self.resnet.maxpool(x_u)
        x_u0 = self.drop(x_u0)

        x_u1 = self.resnet.layer1(x_u)
        x_u1 = self.drop(x_u1)

        x_u2 = self.resnet.layer2(x_u1)
        x_u2 = self.drop(x_u2)

        x_u3 = self.resnet.layer3(x_u2)
        x_u3 = self.drop(x_u3)

        x_u4 = self.resnet.layer4(x_u3)
        x_u4 = self.drop(x_u4)





        x_c4 = self.FF_5(x_u4,score_T_4,score_T_4)
        x_c4_pred = self.final_4(x_c4)
        x_c4_pred = F.interpolate(x_c4_pred, scale_factor=2, mode='bilinear')
        x_d4_pred_s = self.sigmoid(x_c4_pred)
        x_c3 = self.FF_4(x_u3, score_T_3,x_d4_pred_s)
        
        x_d3_pred = self.final_3(x_c3)
        x_d3_pred = F.interpolate(x_d3_pred, scale_factor=2, mode='bilinear')
        x_d3_pred_s = self.sigmoid(x_d3_pred)
        x_c2 = self.FF_3(x_u2, score_T_2,x_d3_pred_s)
       
        x_d2_pred = self.final_2(x_c2)
        x_d2_pred = F.interpolate(x_d2_pred, scale_factor=2, mode='bilinear')
        x_d2_pred_s = self.sigmoid(x_d2_pred)
        x_c1 = self.FF_2(x_u1, score_T_1,x_d2_pred_s)
        
        x_d1_pred = self.final_1(x_c1)
       
        x_d1_pred= F.interpolate(x_d1_pred, scale_factor=2, mode='bilinear')
        x_d1_pred_s = self.sigmoid(x_d1_pred)
        x_d0  = self.RO(x_c1,x_u0,x_d1_pred_s)







        # decoder part
        map_4 = F.interpolate(x_c4_pred, scale_factor=16, mode='bilinear')
        map_3 = F.interpolate(x_d3_pred, scale_factor=8, mode='bilinear')
        map_2 = F.interpolate(x_d2_pred, scale_factor=4, mode='bilinear')
        map_1 = F.interpolate(x_d1_pred, scale_factor=2, mode='bilinear')
        map_0 = F.interpolate(self.final_0(x_d0), scale_factor=2, mode='bilinear')
        return map_4,map_3, map_2, map_1,map_0

    def init_weights(self):
        self.final_0.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final_3.apply(init_weights)
        self.final_4.apply(init_weights)
        self.FF_5.apply(init_weights)
        self.FF_4.apply(init_weights)
        self.FF_3.apply(init_weights)
        self.FF_2.apply(init_weights)
        self.RO.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



