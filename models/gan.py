import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, num_D=3):
        super(Discriminator, self).__init__()
        self.num_D = num_D

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - (i+1)))
            result.append([model(input)])
            if i < (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        self.model = nn.Sequential(
            nn.Linear(input_nc, ndf),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)