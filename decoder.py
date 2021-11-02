# author: yona

import torch
import torch.nn as nn
import torch.nn.functional as F


class decode(torch.nn.Module):
    def __init__(self, input_channel, output_channel, resnet_block_num=20, upsampling=2):
        super().__init__()
        # got idea from resnet18
        img_feat = 80
        ResBLK = []
        for _ in range(resnet_block_num):
            ResBLK += [res_block(input_channel, input_channel)]
        ResBLK_foreground = ResBLK
        ResBLK_alpha = ResBLK

        ResBLK_alpha_output = []
        for _ in range(upsampling):
            ResBLK_alpha_output += [upsampling_block(input_channel, int(input_channel / 2))]
            input_channel = int(input_channel / 2)
        ResBLK_alpha_output += [final_padding_block(input_channel, 1)]

        input_channel = input_channel * 2 * upsampling
        ResBLK_foreground_output = []
        for _ in range(upsampling - 1):
            ResBLK_foreground_output += [upsampling_block(input_channel, int(input_channel / 2))]
            input_channel = int(input_channel / 2)
        F_out = [upsampling_block(input_channel * 2, 64)]
        F_out += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_channel, kernel_size=7, padding=0)]

        self.ResBLK = nn.Sequential(*ResBLK)
        self.ResBLK_foreground = nn.Sequential(*ResBLK_foreground)
        self.ResBLK_alpha = nn.Sequential(*ResBLK_alpha)
        self.ResBLK_alpha_output = nn.Sequential(*ResBLK_alpha_output)
        self.ResBLK_foreground_output = nn.Sequential(*ResBLK_foreground_output)
        self.F_out = nn.Sequential(*F_out)

    def forward(self,comb_feat,img_feat):
        x = self.ResBLK(comb_feat)
        fg = self.ResBLK_foreground(x)
        alpha = self.ResBLK_alpha(x)
        # decoder part
        fg = self.ResBLK_foreground_output(fg)
        alpha = self.ResBLK_alpha_output(alpha)
        # output F and alpha
        F = self.F_out(torch.cat([fg, img_feat], dim=1))
        return F, alpha


class res_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, stride=1, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(output_channel, output_channel, stride=1, padding=1, kernel_size=3)
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
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        outputs = self.upsampling(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(self.bn(outputs))
        return outputs


class final_padding_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.padding = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=7, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(self.padding(x))
        x = F.tanh(x)
        return x



