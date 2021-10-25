import torch
import torch.nn as nn


class CSblock(nn.Module):
    def __init__(self, input_channel, output_channel, num_filter=64, nf_part=64):
        """

        param input_channel: a 1*3 or 1*4(videos) list with 4 integers which represents the sizes of image,
                            background, segmentation and motion
        param num_filter: a integer which represents the number of filters in the generator

        """
        super(CSblock, self).__init__()

        self.in_channel = input_channel
        self.out_channel = output_channel
        self.nf = num_filter
        self.nfp = nf_part

        self.img_encoder1 = nn.Sequential(          # (128, W/2, H/2)
            nn.Conv2d(self.in_channel[0], self.nf, kernel_size=7, padding=3, bias=True),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(),
            nn.Conv2d(self.nf, self.nf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.nf * 2),
            nn.ReLU()
        )
        self.img_encoder2 = nn.Sequential(          # (256, W/4, H/4)
            nn.Conv2d(self.nf * 2, self.nf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU()
        )

        self.bg_encoder = self.encoder(1)           # (256, W/4, H/4)
        self.seg_encoder = self.encoder(2)          # (256, W/4, H/4)
        self.mt_encoder = self.encoder(3)     # (256, W/4, H/4)

        self.comb_back = self.combine_features()    # (64, W/4, H/4)
        self.comb_seg = self.combine_features()     # (64, W/4, H/4)
        self.comb_mt = self.combine_features()      # (64, W/4, H/4)

        self.comb_all = nn.Sequential(              # (256, W/4, H/4)
            nn.Conv2d(self.nf * 4 + self.nfp * 3, self.nf * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU()
        )

    def forward(self, image, background, segmentation, motion):
        """
        param:
            the input image, background, segmentation and motion(videos only) should have same size

        return:
            the return out_feature is a torch with torch size ()

        """

        img_feature128 = self.img_encoder1(image)           # (128, W/2, H/2)
        img_feature256 = self.img_encoder2(img_feature128)  # (256, W/4, H/4)
        bg_feature = self.bg_encoder(background)            # (256, W/4, H/4)
        seg_feature = self.seg_encoder(segmentation)        # (256, W/4, H/4)
        mt_feature = self.mt_encoder(motion)                # (256, W/4, H/4)

        # Selector
        ibg_feature = self.comb_back(torch.cat([img_feature256, bg_feature], dim=1))                   # (64, W/4, H/4)
        ise_feature = self.comb_seg(torch.cat([img_feature256, seg_feature], dim=1))                   # (64, W/4, H/4)
        imt_feature = self.comb_mt(torch.cat([img_feature256, mt_feature], dim=1))                     # (64, W/4, H/4)

        # Combinator
        comb_feature = torch.cat([img_feature256, ibg_feature, ise_feature, imt_feature], dim=1)       # (7*64, W/4, H/4)
        out_feature = self.comb_all(comb_feature)

        return img_feature128, out_feature  # (128, W/4, H/4), (256, W/4, H/4)

    def encoder(self, id_input):
        encoder = nn.Sequential(
            nn.Conv2d(self.in_channel[id_input], self.nf, kernel_size=7, padding=3, bias=True),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(),
            nn.Conv2d(self.nf, self.nf * 4, kernel_size=5, stride=4, padding=2, bias=True),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU()
        )
        return encoder

    def combine_features(self):
        combine_model = nn.Sequential(
            nn.Conv2d(2 * self.nf * 4, self.nfp, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.nfp),
            nn.ReLU()
        )
        return combine_model





