#author: yona
from CSblock import CSblock
from decoder import decode
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Net, self).__init__()
        # encoder part
        self.CSblock = CSblock(input_channel=input_channel, output_channel=output_channel)
        # decoder part
        self.decoder = decode(256, 3)

    def forward(self, image, background, segmentation, motion):
        img_feature128, out_feature = self.CSblock(image, background, segmentation, motion)
        F, alpha = self.decoder(out_feature, img_feature128)
        return F, alpha
