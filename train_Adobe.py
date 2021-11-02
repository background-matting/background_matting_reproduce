#author: yona,Danyang
from origin_data_loader import AdobeBGMData
from model import Net
from torch.utils.data import DataLoader
from loss_function import L1_loss, compose_loss, gradient_loss
from torch.autograd import Variable
from util import show_figure
import torch.optim as optim
import torch
import torch.nn as nn
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

"""Parses arguments."""
parser = argparse.ArgumentParser(description='train first stage with adobe')
parser.add_argument('-bs', '--batch_size', type=int, default=4, help='batch size of the model.')
parser.add_argument('-lr', '--learning_rate', type=int, default=1e-4, help='learning rate of the model.')
parser.add_argument('-np', '--net_path', type=str, help='saved net path.')
parser.add_argument('-op', '--optimizer_path', type=str, help='saved optimizer path.')
args=parser.parse_args()


image_path = 'train'
net_path = args.net_path
optimizer_path = args.optimizer_path
temp_data = AdobeBGMData(trimap_k=[5, 5], resolution=[320, 320], data_path=image_path, noise=True) # one to one
temp_loader = DataLoader(temp_data, batch_size=args.batch_size)
net = Net(input_channel=(3, 3, 1, 4), output_channel=4).cuda()
device_ids = [0, 1]
## initialize
net = torch.nn.DataParallel(net, device_ids=device_ids)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
## resume
#state_dict_load = torch.load(net_path)
#net.load_state_dict(state_dict_load)
#opt_load = torch.load(optimizer_path)
#optimizer.load_state_dict(opt_load)

l1_loss = L1_loss()
c_loss = compose_loss()
g_loss = gradient_loss()
epochs = 60
all_the_loss = []

print('starting training.........')
for epoch in range(epochs):
    each_loss = 0
    for i, data in enumerate(temp_loader):
        fg, bg, alpha, image, seg, bg_tr, mt = data['fg'], data['bg'], data['alpha'], data['image'], data['seg'], data[
            'bg_tr'], data['multi_fr']
        fg, bg, alpha, image, seg, bg_tr, mt = Variable(fg.cuda()), Variable(bg.cuda()), Variable(alpha.cuda()), Variable(image.cuda()), Variable(seg.cuda()), Variable(bg_tr.cuda()), Variable(mt.cuda())
        # find foreground and alpha matte
        F, al = net(image, bg_tr, seg, mt)
        # get loss
        mask = (alpha > -0.99).type(torch.cuda.FloatTensor)
        mask0 = Variable(torch.ones(alpha.shape).cuda())
        al_loss = l1_loss(alpha, al, mask0)
        F_loss = l1_loss(fg, F, mask)
        al_mask = (al > 0.90).type(torch.cuda.FloatTensor)
        F_c = image * al_mask + F * (1 - al_mask)
        F_c_loss = c_loss(image, al, F_c, bg, mask0)
        al_F_c_loss = g_loss(alpha, al, mask0)
        loss = al_loss + 2 * F_loss + F_c_loss + al_F_c_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        each_loss += loss.item()
    one_epoch_loss = each_loss/len(temp_loader)
    print('Train Epoch:', epoch, 'Loss:', one_epoch_loss)
    if len(all_the_loss) !=0 and one_epoch_loss < min(all_the_loss):
        torch.save(net.state_dict(),args.net_path)
        torch.save(optimizer.state_dict(),args.optimizer_path)
    all_the_loss.append(one_epoch_loss)



