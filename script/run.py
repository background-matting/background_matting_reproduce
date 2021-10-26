from models.origin_data_loader import AdobeBGMData
from models.model import Net
from torch.utils.data import DataLoader
from script.loss_function import L1_loss, compose_loss, gradient_loss
from torch.autograd import Variable
import torch.optim as optim
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_path = '../dataset/data'
temp_data = AdobeBGMData(trimap_k=[5, 5], resolution=[320, 320], data_path=image_path, noise=True) # one to one
temp_loader = DataLoader(temp_data, batch_size=len(temp_data))
net = Net(input_channel=(3, 3, 1, 4), output_channel=4)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
l1_loss = L1_loss();
c_loss = compose_loss();
g_loss = gradient_loss()
epochs = 60
all_the_loss = []
net_path = ''
optimizer_path = ''

print('starting training.........')
for epoch in range(epochs):
    for i, data in enumerate(temp_loader):
        fg, bg, alpha, image, seg, bg_tr, mt = data['fg'], data['bg'], data['alpha'], data['image'], data['seg'], data[
            'bg_tr'], data['multi_fr']
        F, al = net(image, bg_tr, seg, mt)
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
        print('Train Epoch:', epoch, 'batch index', i, 'Loss:', loss.item())
        if loss < min(all_the_loss) or len(all_the_loss)==0:
            torch.save(net.state_dict(),net_path)
            torch.save(optimizer.state_dict(),optimizer_path)
        all_the_loss.append(loss)



