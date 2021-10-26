from models.CSblock import CSblock
from models.decoder import decode
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Net, self).__init__()
        self.CSblock = CSblock(input_channel=input_channel, output_channel=output_channel)
        self.decoder = decode(256, 3)

    def forward(self, image, background, segmentation, motion):
        img_feature128, out_feature = self.CSblock(image, background, segmentation, motion)
        F, alpha = self.decoder(img_feature128, out_feature)
        return F, alpha
