#author yona
from util import to_tensor,show_figure
import cv2
import skimage
import skimage.io
from model import Net
from Composition_code import composite4
import torch
import numpy as np
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
device_ids = [0, 1]

parser = argparse.ArgumentParser(description='train first stage with adobe')
parser.add_argument('-np', '--net_path', type=str, help='loaded net path.')
args=parser.parse_args()


net_path = args.net_path
net = Net(input_channel=(3, 3, 1, 4), output_channel=4).cuda()
net = torch.nn.DataParallel(net, device_ids=device_ids)
state_dict_load = torch.load(net_path)
net.load_state_dict(state_dict_load)

img = cv2.imread('one_img/img.png')
img_gray = cv2.imread('one_img/img.png',cv2.IMREAD_GRAYSCALE)
bg = cv2.imread('one_img/background.png')
bg1 = cv2.imread('one_img/bg1.png')
seg = cv2.imread('one_img/seg.png', cv2.IMREAD_GRAYSCALE)
bg1 = cv2.resize(bg1, dsize=(320, 320))

img = cv2.resize(img, dsize=(320, 320))
img_gray = cv2.resize(img_gray, dsize=(320, 320))
bg = cv2.resize(bg, dsize=(320, 320))
bg = skimage.exposure.adjust_gamma(bg, np.random.normal(1, 0.12))
seg = cv2.resize(seg, dsize=(320, 320))

img = to_tensor(img)
bg = to_tensor(bg)
bg1 = to_tensor(bg1)
seg = to_tensor(seg)
img_gray = to_tensor(img_gray)

motion = img_gray.view(1,320,320)
motion = torch.cat([motion,motion,motion,motion],0)

img = img.view(1,3,320,320).cuda()
bg = bg.view(1,3,320,320).cuda()
bg1 = bg1.view(1,3,320,320).cuda()
seg = seg.view(1,1,320,320).cuda()
motion = motion.view(1,4,320,320).cuda()

F, al = net(img, bg, seg, motion)
output_images = composite4(F,bg1,al)
show_figure(output_images[0],'one_img/result.png')