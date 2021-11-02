#author: yona
from origin_data_loader import AdobeBGMData
from model import Net
from torch.utils.data import DataLoader
from loss_function import L1_loss, compose_loss, gradient_loss
from torch.autograd import Variable
from util import show_figure
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import cv2
import os
from Composition_code import composite4
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"


def to_tensor(pic):
    if len(pic.shape) >= 3:
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
    else:
        img = torch.from_numpy(pic)
        img = img.unsqueeze(0)
    # backward compatibility

    return 2 * (img.float().div(255)) - 1


"""Parses arguments."""
parser = argparse.ArgumentParser(description='train first stage with adobe')
parser.add_argument('-np', '--net_path', type=str, help='loaded net path.')
parser.add_argument('-op', '--optimizer_path', type=str, help='loaded optimizer path.')
args = parser.parse_args()

image_path = 'test'
net_path = args.net_path
optimizer_path = args.optimizer_path
temp_data = AdobeBGMData(trimap_k=[5, 5], resolution=[320, 320], data_path=image_path, noise=True)  # one to one
temp_loader = DataLoader(temp_data, batch_size=1)
net = Net(input_channel=(3, 3, 1, 4), output_channel=4).cuda()
device_ids = [0, 1]
## initialize
net = torch.nn.DataParallel(net, device_ids=device_ids)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
## resume
state_dict_load = torch.load(net_path)
net.load_state_dict(state_dict_load)
opt_load = torch.load(optimizer_path)
optimizer.load_state_dict(opt_load)

l1_loss = L1_loss()
c_loss = compose_loss()
g_loss = gradient_loss()
all_the_loss = []
# read black background
black = cv2.imread('black.jpg')
black = to_tensor(black)
black = black.view(1, 3, 320, 320).cuda()

print("start test ........")
for i, data in enumerate(temp_loader):
    fg, bg, alpha, image, seg, bg_tr, mt, original_bg = data['fg'], data['bg'], data['alpha'], data['image'], data[
        'seg'], data[
                                                            'bg_tr'], data['multi_fr'], data['original_bg']
    fg, bg, alpha, image, seg, bg_tr, mt, original_bg = Variable(fg.cuda()), Variable(bg.cuda()), Variable(
        alpha.cuda()), Variable(image.cuda()), Variable(seg.cuda()), Variable(bg_tr.cuda()), Variable(
        mt.cuda()), Variable(original_bg.cuda())
    F, al = net(image, bg_tr, seg, mt)
    output_images = composite4(F, original_bg, al)
    show_figure(output_images[0], "recon_img.png")

    # what following comment did : save other images
    # output_F = composite4(F, black, al)

    # path_output_images = 'result/GAN_output'+str(i)+'.jpg'
    # path_input_images = 'result/GAN_input'+str(i)+'.jpg'
    # path_alpha_images = 'result/GAN_alpha'+str(i)+'.jpg'
    # path_foreground_images = 'result/foreground'+str(i)+'.jpg'
    # show_figure(al[0],"alpha/Adobe10_al.png")
    # show_figure(alpha[0],"alpha/GT.png")
    # torch.save(al,"alpha/Adobe10_al.pt")
    # torch.save(alpha,"alpha/GT.pt")
    # show_figure(output_images[0],path_output_images)
    # show_figure(image[0],path_input_images)
    # show_figure(output_F[0],path_foreground_images)
    # show_figure(al[0],path_alpha_images)
    break
