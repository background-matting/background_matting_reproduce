# author Di
'''
reference: https://github.com/senguptaumd/Background-Matting.git
'''
import torch
import skimage
import skimage.io
import os
import cv2
import random
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

unknown_code = 128


class AdobeBGMData(Dataset):
    def __init__(self, trimap_k, resolution, noise, data_path='./Data', sample_transform=None):
        self.root_path = data_path
        self.img_files = os.listdir(os.path.join(data_path, 'img'))
        self.fg_files = os.listdir(os.path.join(data_path, 'fg'))
        self.bg_files = os.listdir(os.path.join(data_path, 'bg'))
        self.aph_files = os.listdir(os.path.join(data_path, 'aph'))
        self.sample_transform = sample_transform
        self.trimap_k = trimap_k
        self.reso = resolution
        self.noise = noise
        self.transform = self.Adobe_transforms()

    def __len__(self):
        return len(self.img_files)

    def Adobe_transforms(self):
        this_transforms = transforms.Compose([
            transforms.Resize([800, 800]),
            transforms.RandomCrop(random.randrange(500, 800, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(self.reso)
        ])
        return this_transforms

    def __getitem__(self, index):

        # load data
        img = Image.open(os.path.join(self.root_path, 'img', self.img_files[index])).convert('RGB')
        fg = Image.open(os.path.join(self.root_path, 'fg', self.fg_files[index])).convert('RGB')
        bg = Image.open(os.path.join(self.root_path, 'bg', self.bg_files[index])).convert('RGB')
        aph = Image.open(os.path.join(self.root_path, 'aph', self.aph_files[index])).convert('L')
        trimap = generate_trimap(aph, self.trimap_k[0], self.trimap_k[1])

        img = self.transform(img)
        fg = self.transform(fg)
        bg = self.transform(bg)
        aph = self.transform(aph)
        trimap = self.transform(Image.fromarray(trimap))

        img = np.array(img)
        fg = np.array(fg)
        bg = np.array(bg)
        aph = np.array(aph)
        trimap = np.array(trimap)

        # Perturb Background: add Gaussian noise or change gamma
        if self.noise:
            if np.random.random_sample() > 0.5:
                bg_tr = add_noise(bg)
            else:
                bg_tr = skimage.exposure.adjust_gamma(bg, np.random.normal(1, 0.12))

        # Create motion cues: transform foreground and create 4 additional images
        frames = np.zeros((fg.shape[0], fg.shape[1], 4))
        for t in range(0, 4):
            img_tr = generate_add_img(fg, aph, bg)
            frames[..., t] = cv2.cvtColor(img_tr, cv2.COLOR_BGR2GRAY)

        sample = {'image': to_tensor(img), 'fg': to_tensor(fg), 'alpha': to_tensor(aph), 'bg': to_tensor(bg),
                  'trimap': to_tensor(trimap), 'bg_tr': to_tensor(bg_tr), 'seg': to_tensor(generate_seg(aph, trimap)),
                  'multi_fr': to_tensor(frames)}

        if self.sample_transform:
            sample = self.sample_transform(sample)
        return sample


class VideoData(Dataset):
    def __init__(self, csv_file, data_config, transform=None):
        self.frames = pd.read_csv(csv_file, sep=';')
        self.transform = transform
        self.reso = data_config['reso']
        if transform is None:
            self.transform = self.video_transforms()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.frames)

    def video_transforms(self):
        this_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(self.reso)
        ])
        return this_transforms

    def __getitem__(self, idx):
        img = skimage.io.imread(self.frames.iloc[idx, 0])
        back = skimage.io.imread(self.frames.iloc[idx, 1])
        seg = skimage.io.imread(self.frames.iloc[idx, 2])

        fr1 = cv2.cvtColor(skimage.io.imread(self.frames.iloc[idx, 3]), cv2.COLOR_BGR2GRAY)
        fr2 = cv2.cvtColor(skimage.io.imread(self.frames.iloc[idx, 4]), cv2.COLOR_BGR2GRAY)
        fr3 = cv2.cvtColor(skimage.io.imread(self.frames.iloc[idx, 5]), cv2.COLOR_BGR2GRAY)
        fr4 = cv2.cvtColor(skimage.io.imread(self.frames.iloc[idx, 6]), cv2.COLOR_BGR2GRAY)

        back_rnd = skimage.io.imread(self.frames.iloc[idx, 7])

        if np.random.random_sample() > 0.5:
            img = cv2.flip(img, 1)
            seg = cv2.flip(seg, 1)
            back = cv2.flip(back, 1)
            back_rnd = cv2.flip(back_rnd, 1)
            fr1 = cv2.flip(fr1, 1)
            fr2 = cv2.flip(fr2, 1)
            fr3 = cv2.flip(fr3, 1)
            fr4 = cv2.flip(fr4, 1)

        # make frames together
        multi_fr = np.zeros((img.shape[0], img.shape[1], 4))
        multi_fr[..., 0] = fr1
        multi_fr[..., 1] = fr2
        multi_fr[..., 2] = fr3
        multi_fr[..., 3] = fr4

        # allow random cropping centered on the segmentation map
        bbox = create_bbox(seg, seg.shape[0], seg.shape[1])
        img = apply_crop(img, bbox, self.reso)
        seg = apply_crop(seg, bbox, self.reso)
        back = apply_crop(back, bbox, self.reso)
        back_rnd = apply_crop(back_rnd, bbox, self.reso)
        multi_fr = apply_crop(multi_fr, bbox, self.reso)

        sample = {'image': to_tensor(img), 'seg': to_tensor(create_seg_guide(seg, self.reso)),
                  'bg': to_tensor(back), 'multi_fr': to_tensor(multi_fr), 'seg-gt': to_tensor(seg),
                  'back-rnd': to_tensor(back_rnd)}

        if self.transform:
            sample = self.transform(sample)
        return sample


def generate_trimap(alpha, K1, K2):
    """
    param alpha: alpha matte image
    param K1, K2: integers which represent the parameter of erode process
    return trimap: trimap image generated by alpha matte

    This function generate trimap based on alpha matte
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    K = np.round((K1 + K2) / 2).astype('int')

    fg = cv2.erode(fg, kernel, iterations=K)
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=2 * K)
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)


def add_noise(bg, sigma=np.random.randint(low=2, high=6), mu=np.random.randint(low=0, high=14) - 7):
    """
    param bg: a background image
    return bg_tr: the background image after adding Gaussian noise

    This function add Gaussian noise to background image to simulate real situation
    """
    w, h, c = bg.shape
    noise = np.random.normal(mu, sigma, (w, h, c))
    bg_tr = bg.astype(np.float32) + noise.reshape((w, h, c))

    bg_tr[bg_tr < 0] = 0
    bg_tr[bg_tr > 255] = 255

    return bg_tr.astype(np.uint8)


def generate_add_img(fg, aph, bg):
    """
    param fg: a foreground image with channel 3
    param aph: an alpha matte image with channel 1
    param bg: a background image with channel 3
    return img: a image generated by moving foreground image of background image

    This function generate the additional images to simulate motion cues in videos
    """

    T = np.random.normal(0, 5, (2, 1))
    theta = np.random.normal(0, 7)
    R = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                  [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
    sc = np.array([[1 + np.random.normal(0, 0.05), 0], [0, 1]])
    sh = np.array([[1, np.random.normal(0, 0.05) * (np.random.random_sample() > 0.5)],
                   [np.random.normal(0, 0.05) * (np.random.random_sample() > 0.5), 1]])
    A = np.concatenate((sc * sh * R, T), axis=1)

    fg_tr = cv2.warpAffine(fg.astype(np.uint8), A, (fg.shape[1], fg.shape[0]), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    aph_tr = cv2.warpAffine(aph.astype(np.uint8), A, (fg.shape[1], fg.shape[0]), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REFLECT)

    bg_tr = add_noise(bg)

    fg_tr = fg_tr.astype(np.float32)
    bg_tr = bg_tr.astype(np.float32)
    aph_tr = np.expand_dims(aph_tr.astype(np.float32) / 255, axis=2)
    img = aph_tr * fg_tr + (1 - aph_tr) * bg_tr
    img = img.astype(np.uint8)

    return img


def to_tensor(img):
    """
    param img: a numpy image
    return img_tensor: a tensor image with (-1, 1)

    This function transform image type from numpy array to tensor
    which can be used in training process.
    """
    if len(img.shape) >= 3:
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))
    else:
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = 2 * (img_tensor.float().div(255)) - 1

    return img_tensor


def generate_seg(alpha, trimap):
    """
    param alpha: an alpha image with channel 1
    param trimap: a trimap image with channel 1
    return seg: a soft segmentation with channel 1

    This function generate soft segmentation based on alpha matte and trimap, and then erode (10-20 steps),
    dilate (15-30 steps) and blur(σ ∈ [3, 5, 7]) the result.
    """

    num_holes = np.random.randint(low=0, high=3)
    crop_size_list = [(15, 15), (25, 25), (35, 35), (45, 45)]
    kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    seg = (alpha > 0.5).astype(np.float32)
    seg = cv2.erode(seg, kernel_er, iterations=np.random.randint(low=10, high=20))
    seg = cv2.dilate(seg, kernel_dil, iterations=np.random.randint(low=15, high=30))

    seg = seg.astype(np.float32)
    seg = (255 * seg).astype(np.uint8)
    for i in range(num_holes):
        x, y = 0, 0
        crop_size = random.choice(crop_size_list)
        h, w = trimap.shape[0:2]
        crop_h, crop_w = crop_size
        val_idx = np.zeros((h, w))
        val_idx[int(crop_h / 2):int(h - crop_h / 2), int(crop_w / 2):int(w - crop_w / 2)] = 1
        y_i, x_i = np.where(np.logical_and(trimap == unknown_code, val_idx == 1))
        num_unknowns = len(y_i)

        if num_unknowns > 0:
            ix = np.random.choice(range(num_unknowns))
            center_x = x_i[ix]
            center_y = y_i[ix]
            x = max(0, center_x - int(crop_w / 2))
            y = max(0, center_y - int(crop_h / 2))

        seg[y: y + crop_h, x: x + crop_w] = 0
        trimap[y: y + crop_h, x: x + crop_w] = 0

    k_size_list = [(21, 21), (31, 31), (41, 41)]
    seg = cv2.GaussianBlur(seg.astype(np.float32), random.choice(k_size_list), 0)
    return seg.astype(np.uint8)


def create_seg_guide(rcnn, reso):
    kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rcnn = rcnn.astype(np.float32) / 255
    rcnn[rcnn > 0.2] = 1
    K = 25

    zero_id = np.nonzero(np.sum(rcnn, axis=1) == 0)
    del_id = zero_id[0][zero_id[0] > 250]
    if len(del_id) > 0:
        del_id = [del_id[0] - 2, del_id[0] - 1, *del_id]
        rcnn = np.delete(rcnn, del_id, 0)
    rcnn = cv2.copyMakeBorder(rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)

    rcnn = cv2.erode(rcnn, kernel_er, iterations=np.random.randint(10, 20))
    rcnn = cv2.dilate(rcnn, kernel_dil, iterations=np.random.randint(3, 7))
    k_size_list = [(21, 21), (31, 31), (41, 41)]
    rcnn = cv2.GaussianBlur(rcnn.astype(np.float32), random.choice(k_size_list), 0)
    rcnn = (255 * rcnn).astype(np.uint8)
    rcnn = np.delete(rcnn, range(reso[0], reso[0] + K), 0)

    return rcnn


def apply_crop(img, bbox, reso):
    img_crop = img[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], ...]
    img_crop = cv2.resize(img_crop, reso)
    return img_crop


def create_bbox(mask, R, C):
    where = np.array(np.where(mask))
    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)

    w = np.maximum(y2 - y1, x2 - x1)
    bd = np.random.uniform(0.1, 0.4)
    x1 = x1 - np.round(bd * w)
    y1 = y1 - np.round(bd * w)
    y2 = y2 + np.round(bd * w)

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if y2 >= C: y2 = C
    if x2 >= R: x2 = R - 1

    bbox = np.around([x1, y1, x2 - x1, y2 - y1]).astype('int')

    return bbox
