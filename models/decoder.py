#modified by yona

import torch
import torch.nn as nn
import torch.nn.functional as F

dim = 1
Block_number = 10

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ResBLK1 = []
        ResBLK2_foreground = []
        ResBLK3_alpha =[]
        for _ in range(Block_number):
            ResBLK1 += res_block()
            ResBLK2_foreground += res_block()
            ResBLK3_alpha += res_block()
        upsampling_foreground =


    def forward(self,x):
        for _ in range(Block_number):
            x = self.ResBLK(x)
        # foreground decode
        for _ in range(Block_number):
            x = self.ResBLK(x)
        # foreground decode
        pass

class res_block():
    def __init__(self):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3)

    def forward(self, x):
        out = F.relu(nn.BatchNorm2d(self.conv1(x)))
        out = nn.BatchNorm2d(self.fc2(out))
        out = out+x
        return out

