# author: yona
import cv2
import numpy as np
import torch


def show_figure(img, path="result/img.jpg"):
    img = (img + 1) / 2 * 255
    img = img.cpu().detach().numpy()
    img = img.transpose((1, 2, 0))
    cv2.imwrite(path, img)

def show_figure_al(img, path="result/img.jpg"):
    img = (img + 1) / 2 * 255
    img = img.squeeze(0)
    img = img.cpu().detach().numpy()
    cv2.imwrite(path, img)

def to_tensor(pic):
    if len(pic.shape) >= 3:
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
    else:
        img = torch.from_numpy(pic)
        img = img.unsqueeze(0)
    # backward compatibility
    return 2 * (img.float().div(255)) - 1