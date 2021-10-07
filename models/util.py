#modified by yona

import torch


def generate_img(F,B):
    result = torch.cat([F, B], dim=1)



