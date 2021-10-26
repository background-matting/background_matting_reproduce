from models.CSblock import CSblock
from models.decoder import decode
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_channel, output_channel, num_filter=64, nf_part=64):
        super(Net, self).__init__()
        self.CSblock = CSblock(input_channel=(3, 3, 1, 4), output_channel=4)
        self.decoder = decode(256, 4)

    def forward(self, image, background, segmentation, motion):
        img_feature128, out_feature = self.CSblock(image, background, segmentation, motion)
        F, alpha = self.decoder(img_feature128, out_feature)
        return F, alpha
