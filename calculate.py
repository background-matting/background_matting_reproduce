# author: Di and yona
import cv2
import numpy as np
import torch
import argparse


def calculate_iou(pred_img, gt_img):
    """
    params pred_img: grayscale image
    params gt_img: grayscale image


    """
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    cross = (gt_img == pred_img).sum()
    iou = cross / (h*w)
    print('The IoU is: ' + str(iou))
    return iou


def calculate_sad(pred_img, gt_img):
    pred_img = pred_img.reshape(-1, 1)
    gt_img = gt_img.reshape(-1, 1)
    dif_img = gt_img/255 - pred_img/255
    dif_img = abs(dif_img)
    sad = sum(dif_img)
    print('The SAD is: ' + str(sad))
    return sad


def calculate_mse(pred_img, gt_img):
    pred_img = pred_img.reshape(-1, 1)
    gt_img = gt_img.reshape(-1, 1)
    dif_img = (gt_img/255 - pred_img/255)**2
    mse = np.mean(dif_img)
    print('The MSE is: ' + str(mse))
    return mse


def aline_size(pred_img, gt_img):
    w = pred_img.shape[0]
    h = pred_img.shape[1]
    gt_img = cv2.resize(gt_img, (h, w), interpolation=cv2.INTER_NEAREST)

    return pred_img, gt_img


"""Parses arguments."""
parser = argparse.ArgumentParser(description='Find MSE, SAD and IoU result.')
parser.add_argument('-op', '--our_path', type=str, help='Input the path of our alpha image.')
parser.add_argument('-gt', '--ground_truth_path', type=str, help='Input the path of groundtruth alpha image.')
parser.add_argument('-ops', '--our_path_save', type=str, help='our alpha image saved as pt.')
parser.add_argument('-gts', '--ground_truth_path_save', type=str, help='groundtruth alpha image saved as pt.')
args=parser.parse_args()

# from png to pt
pred_aph_path = args.our_path
img = cv2.cvtColor(cv2.imread(pred_aph_path), cv2.COLOR_BGR2GRAY)
torch.save(img, args.our_path_save)
gt_aph_path = args.ground_truth_path
img = cv2.cvtColor(cv2.imread(gt_aph_path), cv2.COLOR_BGR2GRAY)
torch.save(img, args.ground_truth_path_save)

# load pt directly
pred_aph = torch.load(args.our_path_save)
gt_aph = torch.load(args.ground_truth_path_save)
pred_aph, gt_aph = aline_size(pred_aph, gt_aph)


iou = calculate_iou(pred_aph, gt_aph)
sad = calculate_sad(pred_aph, gt_aph)
mse = calculate_mse(pred_aph, gt_aph)


