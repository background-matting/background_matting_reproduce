# modified by yona

import torch
import torch.nn as nn
import torch.nn.functional as F


class decode(torch.nn.Module):
    def __init__(self, input_channel, output_channel, resnet_block_num=20):
        super().__init__()
        self.preprocess = nn.sequence(nn.Conv2d(input_channel, output_channel, stride=1),
                                      nn.BatchNorm2d(output_channel),
                                      nn.ReLU(inplace=True))
        self.ResBLK = []
        for _ in resnet_block_num:
            self.ResBLK += res_block(output_channel, output_channel)
        self.ResBLK_foreground = self.ResBLK
        self.ResBLK_alpha = self.ResBLK

        self.ResBLK_foreground_output = upsampling_block()
        self.ResBLK_alpha_output = self.ResBLK_foreground_output

        self.F_out = upsampling_block()

    def forward(self, img_feat, comb_feat):
        out_all = torch.cat([img_feat, comb_feat], dim=1)
        out_all = self.preprocess(out_all)
        x = self.ResBLK(out_all)
        fg = self.ResBLK_foreground(x)
        alpha = self.ResBLK_alpha(x)
        fg = self.ResBLK_foreground_output(fg)
        alpha = self.ResBLK_alpha_output(alpha)
        F = self.F_out(torch.cat([fg, img_feat], dim=1))
        return F, alpha


class res_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        self.conv1  = nn.Conv2d(input_channel, output_channel, stride=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, stride=1),
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = F.relu(self.bn(outputs))
        outputs = self.conv2(outputs)
        outputs = self.bn(outputs)
        outputs = outputs + inputs
        return outputs


class upsampling_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channel*2, output_channel, stride=1)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        outputs = self.upsampling(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(self.bn(outputs))
        return outputs

