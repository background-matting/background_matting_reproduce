#author: yona
import cv2
import os
import numpy as np
import torch

def to_tensor(pic):
    if len(pic.shape) >= 3:
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
    else:
        img = torch.from_numpy(pic)
        img = img.unsqueeze(0)

    return 2 * (img.float().div(255)) - 1

black = np.zeros((320,320))
cv2.imwrite('black.jpg', black)
black = cv2.imread('black.jpg')
black = to_tensor(black)
black = black.view(1,3,320,320)