import torch
import skimage
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import os
import cv2
from PIL import Image
from torchvision import transforms
import random

unknown_code = 128

this_transforms = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.RandomCrop(random.randrange(500, 800, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
])


# def generate_seg()


class AdobeBGMData(Dataset):
    def __init__(self, trimap_k, resolution, noise, data_path='./Data', transform=None):
        self.root_path = data_path
        self.img_files = os.listdir(os.path.join(data_path, 'img'))
        self.fg_files = os.listdir(os.path.join(data_path, 'fg'))
        self.bg_files = os.listdir(os.path.join(data_path, 'bg'))
        self.aph_files = os.listdir(os.path.join(data_path, 'aph'))
        self.transform = transform
        self.trimap_k = trimap_k
        self.reso = resolution
        self.noise = noise

        if transform is None:
            self.transform = this_transforms
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        # load data
        img = Image.open(os.path.join(self.root_path, 'img', self.img_files[index])).convert('RGB')
        fg = Image.open(os.path.join(self.root_path, 'fg', self.fg_files[index])).convert('RGB')
        bg = Image.open(os.path.join(self.root_path, 'bg', self.bg_files[index])).convert('RGB')
        aph = Image.open(os.path.join(self.root_path, 'aph', self.aph_files[index])).convert('L')

        fg = cv2.resize(fg, dsize=(800, 800))
        aph = cv2.resize(aph, dsize=(800, 800))
        bg = cv2.resize(bg, dsize=(800, 800))
        img = cv2.resize(img, dsize=(800, 800))

        # random flip
        if np.random.random_sample() > 0.5:
            img = cv2.flip(img, 1)
            fg = cv2.flip(fg, 1)
            bg = cv2.flip(bg, 1)
            aph = cv2.flip(aph, 1)

        trimap = generate_trimap(aph, self.trimap_k[0], self.trimap_k[1], False)

        # random crop+scale
        different_sizes = [(576, 576), (608, 608), (640, 640), (672, 672), (704, 704), (736, 736), (768, 768),
                           (800, 800)]
        crop_size = random.choice(different_sizes)

        x, y = random_choice(trimap, crop_size)

        img = safe_crop(img, x, y, crop_size, self.reso)
        fg = safe_crop(fg, x, y, crop_size, self.reso)
        aph = safe_crop(aph, x, y, crop_size, self.reso)
        bg = safe_crop(bg, x, y, crop_size, self.reso)
        trimap = safe_crop(trimap, x, y, crop_size, self.reso)

        # Perturb Background: random noise addition or gamma change
        if self.noise:
            if np.random.random_sample() > 0.6:
                sigma = np.random.randint(low=2, high=6)
                mu = np.random.randint(low=0, high=14) - 7
                bg_tr = add_noise(bg, mu, sigma)
            else:
                bg_tr = skimage.exposure.adjust_gamma(bg, np.random.normal(1, 0.12))

        # Create motion cues: transform foreground and create 4 additional images
        affine_fr = np.zeros((fg.shape[0], fg.shape[1], 4))
        for t in range(0, 4):
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
            alpha_tr = cv2.warpAffine(aph.astype(np.uint8), A, (fg.shape[1], fg.shape[0]), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT)

            sigma = np.random.randint(low=2, high=6)
            mu = np.random.randint(low=0, high=14) - 7
            bg_tr0 = add_noise(bg, mu, sigma)

            affine_fr[..., t] = cv2.cvtColor(composite(fg_tr, bg_tr0, alpha_tr), cv2.COLOR_BGR2GRAY)

        sample = {'image': to_tensor(img), 'fg': to_tensor(fg), 'alpha': to_tensor(aph), 'bg': to_tensor(bg),
                  'trimap': to_tensor(trimap), 'bg_tr': to_tensor(bg_tr), 'seg': to_tensor(create_seg(aph, trimap)),
                  'multi_fr': to_tensor(affine_fr)}

        # sample = {'image': to_tensor(img), 'fg': to_tensor(fg), 'alpha': to_tensor(aph), 'bg': to_tensor(bg),
        #           'trimap': to_tensor(trimap), 'bg_tr': to_tensor(bg_tr), 'seg': to_tensor(create_seg(aph, trimap))}

        if self.transform:
            sample = self.transform(sample)
        return sample


def generate_trimap(alpha, K1, K2, train_mode):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    if train_mode:
        K = np.random.randint(K1, K2)
    else:
        K = np.round((K1 + K2) / 2).astype('int')

    fg = cv2.erode(fg, kernel, iterations=K)
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=2 * K)
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)


def random_choice(trimap, crop_size=(320, 320)):
    img_height, img_width = trimap.shape[0:2]
    crop_height, crop_width = crop_size

    val_idx = np.zeros((img_height, img_width))
    val_idx[int(crop_height / 2):int(img_height - crop_height / 2),
    int(crop_width / 2):int(img_width - crop_width / 2)] = 1
    y_indices, x_indices = np.where(np.logical_and(trimap == unknown_code, val_idx == 1))
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))

    return x, y


def safe_crop(mat, x, y, crop_size, img_size, cubic=True):
    img_rows, img_cols = img_size
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (img_rows, img_cols):
        if cubic:
            ret = cv2.resize(ret, dsize=(img_rows, img_cols))
        else:
            ret = cv2.resize(ret, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
    return ret


def add_noise(back, mean, sigma):
    back = back.astype(np.float32)
    row, col, ch = back.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    # gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
    noisy = back + gauss

    noisy[noisy < 0] = 0;
    noisy[noisy > 255] = 255;

    return noisy.astype(np.uint8)


def to_tensor(pic):
    if len(pic.shape) >= 3:
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
    else:
        img = torch.from_numpy(pic)
        img = img.unsqueeze(0)
    # backward compatibility

    return 2 * (img.float().div(255)) - 1


def composite(fg, bg, a):
    fg = fg.astype(np.float32)
    bg = bg.astype(np.float32)
    a = a.astype(np.float32)
    alpha = np.expand_dims(a / 255, axis=2)
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im


def create_seg(alpha, trimap):
    # old
    num_holes = np.random.randint(low=0, high=3)
    crop_size_list = [(15, 15), (25, 25), (35, 35), (45, 45)]
    kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seg = (alpha > 0.5).astype(np.float32)
    # print('Before %.4f max: %.4f' %(seg.sum(),seg.max()))
    # old
    seg = cv2.erode(seg, kernel_er, iterations=np.random.randint(low=10, high=20))
    seg = cv2.dilate(seg, kernel_dil, iterations=np.random.randint(low=15, high=30))
    # print('After %.4f max: %.4f' %(seg.sum(),seg.max()))
    seg = seg.astype(np.float32)
    seg = (255 * seg).astype(np.uint8)
    for i in range(num_holes):
        crop_size = random.choice(crop_size_list)
        cx, cy = random_choice(trimap, crop_size)
        seg = crop_holes(seg, cx, cy, crop_size)
        trimap = crop_holes(trimap, cx, cy, crop_size)
    k_size_list = [(21, 21), (31, 31), (41, 41)]
    seg = cv2.GaussianBlur(seg.astype(np.float32), random.choice(k_size_list), 0)
    return seg.astype(np.uint8)


def crop_holes(img, cx, cy, crop_size):
    img[cy:cy + crop_size[0], cx:cx + crop_size[1]] = 0
    return img
