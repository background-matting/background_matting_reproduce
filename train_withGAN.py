#author yona and Danyang
import torch.optim as optim
import torch
import torch.nn as nn
import os
import numpy as np
from origin_data_loader import AdobeBGMData
from model import Net
from discriminator import Discriminator
from torch.utils.data import DataLoader
from origin_data_loader import VideoData
from loss_function import L1_loss, compose_loss, gradient_loss, GANloss
from torch.autograd import Variable
from Composition_code import composite4
from util import to_tensor
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,2,1,0"

net_path = 'net/net_10.pt'
netG_path = 'net/netG.pt'
optimizerG_path = 'net/optG.pt'
netD_path = 'net/netD.pt'
optimizerD_path = 'net/optD.pt'
batch_size = 8
# temp_data = AdobeBGMData(data_path=image_path, trimap_k=[5, 5], resolution=[320, 320], noise=True) # one to one
temp_data = VideoData(csv_file='Video_data_train.csv', data_config={'reso': (320, 320)})
temp_loader = DataLoader(temp_data, batch_size=batch_size)
## initial 
#load Adobe net 
net = Net(input_channel=(3, 3, 1, 4), output_channel=4).cuda()
device_ids = [0, 1, 2, 3]
net = torch.nn.DataParallel(net, device_ids=device_ids)
state_dict_load = torch.load(net_path)
net.load_state_dict(state_dict_load)
net.eval()
#freeze adobe
for param in net.parameters():
	param.requires_grad = False
#initial real net
netG = Net(input_channel=(3, 3, 1, 4), output_channel=4).cuda()
netG = torch.nn.DataParallel(netG, device_ids=device_ids)
optimizerG = optim.Adam(netG.parameters(), lr=1e-4)
#initial Discriminator
netD = Discriminator(input_nc=3,ndf=64,n_layers=3, norm_layer=nn.BatchNorm2d, num_D=1).cuda()
netD = torch.nn.DataParallel(netD, device_ids=device_ids)
optimizerD = optim.Adam(netD.parameters(), lr=1e-4)
# loss
l1_loss = L1_loss()
c_loss = compose_loss()
g_loss = gradient_loss()
gan_loss = GANloss()
epochs = 50
wt = 1
all_the_loss = []
# read bg_hat
bg_hat_list = sorted(os.listdir('bghat'))

print('starting training.........')
for epoch in range(epochs):
    each_epoch_gloss = 0
    each_epoch_dloss = 0
    netG.train()
    netD.train()
    for i, data in enumerate(temp_loader):
        image, seg, bg, mt, seg_gt, back_rnd = data['image'], data['seg'], data['bg'], data['multi_fr'], data['seg-gt'], data['back-rnd']
        image, seg, bg, mt, seg_gt, back_rnd = Variable(image.cuda()), Variable(seg.cuda()), Variable(bg.cuda()), Variable(mt.cuda()), Variable(seg_gt.cuda()), Variable(back_rnd.cuda())
        # fg, bg, alpha, image, seg, bg_tr, mt = data['fg'], data['bg'], data['alpha'], data['image'], data['seg'], data[
        #     'bg_tr'], data['multi_fr']
        # fg, bg, alpha, image, seg, bg_tr, mt = Variable(fg.cuda()), Variable(bg.cuda()), Variable(
        #     alpha.cuda()), Variable(image.cuda()), Variable(seg.cuda()), Variable(bg_tr.cuda()), Variable(
        #     mt.cuda())
        bg_hat = None
        for j in range(batch_size):
            bg_h = cv2.imread(os.path.join('background/',bg_hat_list[j*batch_size]))
            bg_h = cv2.resize(bg_h, dsize=(320, 320))
            bg_h = to_tensor(bg_h)
            bg_h = bg_h.view(1,3,320,320)
            if bg_hat != None:
                bg_hat = torch.cat((bg_hat,bg_h),0)
            else:
                bg_hat = bg_h
        bg_hat = bg_hat.cuda()
        adobe_F, adobe_al = net(image, bg, seg, mt)
        real_F,real_al = netG(image, bg, seg, mt)

        mask = (real_al > -0.99).type(torch.cuda.FloatTensor)
        mask0 = Variable(torch.ones(real_al.shape).cuda())
        mask1 = (seg_gt > -0.95).type(torch.cuda.FloatTensor)
        al_mask = (real_al > 0.90).type(torch.cuda.FloatTensor)

        # loss
        al_loss = l1_loss(adobe_al, real_al, mask0)
        F_loss = l1_loss(adobe_F, real_F, mask)
        # F_c = image * al_mask + F * (1 - al_mask)
        F_c_loss = c_loss(image, real_al, real_F, bg, mask1)
        # permute = torch.LongTensor(np.random.permutation(bg.shape[0]))
        # perm_bg = bg[permute,:,:,:]
        image_com = composite4(real_F,bg_hat,real_al)
        

        fake_response = netD(image_com)

        GAN_loss = gan_loss(fake_response,label_type=True)
        lossG = GAN_loss + wt*0.05*(F_c_loss+al_loss+F_loss)

        optimizerG.zero_grad()
        lossG.backward()
        optimizerG.step()

        # discriminator
        real_response = netD(image)
        real_GAN_loss = gan_loss(real_response, label_type=True)
        lossD = 0.5*(GAN_loss.detach()+real_GAN_loss)
        if i%5 == 0:
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
        each_epoch_gloss += lossG.item()
        each_epoch_dloss += lossD.item()
        print("   ","batch_idx: ",i)
    one_epoch_lossG = each_epoch_gloss/len(temp_loader)
    one_epoch_lossD = each_epoch_dloss/len(temp_loader)
    print('Train Epoch:', epoch, 'GLoss:', one_epoch_lossG, 'DLoss:', one_epoch_lossD)
    if len(all_the_loss) !=0 and one_epoch_lossG < min(all_the_loss):
        torch.save(netG.state_dict(),netG_path)
        torch.save(netD.state_dict(),netD_path)
        torch.save(optimizerG.state_dict(),optimizerG_path)
        torch.save(optimizerD.state_dict(),optimizerD_path)
    all_the_loss.append(one_epoch_lossG)
    if (epoch%2 == 0):
        wt=wt/2



