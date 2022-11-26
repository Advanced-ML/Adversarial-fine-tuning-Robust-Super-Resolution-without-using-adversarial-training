import torch
import torch.nn as nn



def MSE_loss(img_lr, img_gt, crop_border=2):
    img_lr = torch.permute(img_lr, (0,3,2,1)) * 255.
    img_gt = torch.permute(img_gt, (0,3,2,1)) * 255.
    if crop_border != 0:
        img_lr = img_lr[:,crop_border:-crop_border, crop_border:-crop_border, ...]
        img_gt = img_gt[:,crop_border:-crop_border, crop_border:-crop_border, ...]
    se = (img_lr - img_gt)**2
    mse = torch.mean(se.reshape(se.shape[0], -1), dim=1)
    mse = torch.sum(mse)
    #print(20. * torch.log10(255. / torch.sqrt(mse)))
    return mse

#----------------------------------------Losses-personalizadas-----------------------