##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.
# author: Danyang

##############################################################
#Set your paths here

#path to provided foreground images
import torch

fg_path = 'fg/'

#path to provided alpha mattes
a_path = 'mask/'

#Path to background images (MSCOCO)
bg_path = 'bg/'

#Path to folder where you want the composited images to go
out_path = 'merged/'

##############################################################

from PIL import Image
import os 
import math
import time

def composite4(fg, bg, a):

    image_com = torch.zeros(fg.shape).cuda()
    for t in range(fg.shape[0]):

        for y in range(a.shape[3]):
            for x in range(a.shape[2]):
                alpha = a[t,:,x,y]
                if alpha >= 1:
                    image_com[t,:,x,y] = fg[t,:,x,y]
                elif alpha <= 0:
                    image_com[t, :, x, y] = bg[t,:,x,y]
                else:
                    image_com[t, :, x, y] = alpha * fg[t, :, x, y] + (1 - alpha) * bg[t, :, x, y]
    return image_com

# num_bgs = 100
#
# fg_files = os.listdir(fg_path)
# a_files = os.listdir(a_path)
# bg_files = os.listdir(bg_path)
#
# bg_iter = iter(bg_files)
# for im_name in fg_files:
#
#     im = Image.open(fg_path + im_name);
#     a = Image.open(a_path + im_name);
#     bbox = im.size
#     w = bbox[0]
#     h = bbox[1]
#
#     if im.mode != 'RGB' and im.mode != 'RGBA':
#         im = im.convert('RGB')
#
#     bcount = 0
#     for i in range(num_bgs):
#
#         bg_name = next(bg_iter)
#         bg = Image.open(bg_path + bg_name)
#         if bg.mode != 'RGB':
#             bg = bg.convert('RGB')
#
#         bg_bbox = bg.size
#         bw = bg_bbox[0]
#         bh = bg_bbox[1]
#         wratio = w / bw
#         hratio = h / bh
#         ratio = wratio if wratio > hratio else hratio
#         if ratio > 1:
#             bg = bg.resize((math.ceil(bw*ratio),math.ceil(bh*ratio)), Image.BICUBIC)
#
#         out = composite4(im, bg, a, w, h)
#
#         out.save(out_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png', "PNG")
#
#         bcount += 1



