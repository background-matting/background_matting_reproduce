import torch
# import data_loader
import origin_data_loader
from CSblock import CSblock
from torch.utils.data import DataLoader

temp_data = origin_data_loader.AdobeBGMData(trimap_k=[5, 5], resolution=[320, 320], noise=True)
temp_loader = DataLoader(temp_data, batch_size=len(temp_data))
for i, data in enumerate(temp_loader):
    # train_loader = DataLoader(data_train, batch_size=4)
    fg, bg, alpha, image, seg, bg_tr, mt = data['fg'], data['bg'], data['alpha'], data['image'], data['seg'], data['bg_tr'], data['multi_fr']

    model = CSblock(input_channel=(3, 3, 1, 4),  output_channel=4)
    print(seg[0].shape)

    img_feature128, out_feature = model(image, bg_tr, seg, mt)

    print(type(img_feature128))

    torch.save(out_feature, './out_feature.pth')
    torch.save(img_feature128, './img_feature128.pth')

    print(out_feature.shape)        # (num_img, 256, W/4, H/4)
    print(img_feature128.shape)     # (num_img, 128, W/2, H/2)
