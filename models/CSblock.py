import torch
import torch.nn as nn

class CSblock(nn.Module):
    def __init__(self, input_channel, output_channel, num_filter=64, nf_part=64):  # nf_part是什么
        # ngf number of filters in the generator
        # input_channel contains the number of input channel of (I,B',S,M)

        self.in_channel = input_channel
        self.out_channel = output_channel
        self.nf = num_filter
        self.nfp = nf_part

        # encoder, with output (256, W/4, H/4)
        self.img_encoder = self.encoder_model(0)
        self.bg_encoder = self.encoder_model(1)
        self.seg_encoder = self.encoder_model(2)
        self.mt_encoder = self.encoder_model(3)

        # combination, with output (64, W/4, H/4)
        self.comb_back = self.combine_features()
        self.comb_seg = self.combine_features()
        self.comb_mt = self.combine_features()

        self.comb_all = nn.Sequential(
            nn.Conv2d(7*self.nf, 4*self.nfp, kernel_size=1, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(4*self.nfp),
            nn.ReLU()
        )

    def forward(self, image, background, segmentation, motion):
        """
        param:
            the input image, background, segmentation and motion(videos only) should have same size

        return:
            the return out_feature is a torch with torch size ()

        """
        img_feature = self.img_encoder(image)         # (256, W/4, H/4)
        bg_feature = self.bg_encoder(background)      # (256, W/4, H/4)
        seg_feature = self.seg_encoder(segmentation)  # (256, W/4, H/4)
        mt_feature = self.mt_encoder(motion)          # (256, W/4, H/4)

        ibg_feature = self.comb_back(torch.cat([img_feature, bg_feature], dim=1))               # (64, W/4, H/4)
        ise_feature = self.comb_seg(torch.cat([img_feature, seg_feature], dim=1))               # (64, W/4, H/4)
        imt_feature = self.comb_mt(torch.cat([img_feature, mt_feature], dim=1))                 # (64, W/4, H/4)
        comb_feature = torch.cat([img_feature, ibg_feature, ise_feature, imt_feature], dim=1)   # (7*64, W/4, H/4)
        out_feature = self.comb_all(comb_feature)                                               # (256, W/4, H/4)

        return out_feature  # (256, W/4, H/4)

    def encoder_model(self, id_input):
        encoder = nn.Sequential(
            nn.Conv2d(self.in_channel[id_input], self.nf, kernel_size=7, padding=3, bias=True),  # 源码padding用的reflect
            nn.BatchNorm2d(self.nf),
            nn.ReLU(),
            nn.Conv2d(self.nf, self.nf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.nf*2),
            nn.ReLU(),
            nn.Conv2d(self.nf * 2, self.nf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU()
        )
        return encoder

    def combine_features(self):
        combine_model = nn.Sequential(
            nn.Conv2d(2*self.nf*4, self.nfp, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.nfp),
            nn.ReLU()
        )
        return combine_model

