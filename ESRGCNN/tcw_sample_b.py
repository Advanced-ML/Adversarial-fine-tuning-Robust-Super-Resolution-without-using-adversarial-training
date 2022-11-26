import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import skimage.measure as measure #tcw201904101622tcw
from torch.autograd import Variable
from ESRGCNN.dataset import TestDataset
from PIL import Image
import cv2 #201904111751tcwi
from torchsummary import summary #tcw20190623
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
#from torchsummaryX import summary #tcw20190625
os.environ['CUDA_VISIBLE_DEVICES']='2'

def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def psnr(im1, im2): #tcw201904101621
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr
#tcw20190413043
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def rgb2ycbcr(img, only_y=True):
    '''
    same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def sample(net, device, dataset):
    scale = 2
    mean_psnr2 = 0
    mean_ssim = 0 #tcw20190413047
    for step, (hr, lr, name) in enumerate(dataset):
        t1 = time.time()
        lr = lr.unsqueeze(0).to(device)
        sr = net(lr, 2).detach().squeeze(0) #detach() break the reversed transformation.
        #print sr.size() #(3,1024,1024)
        lr = lr.squeeze(0)
        #print lr.size() #(3,512,512)
        t2 = time.time()

        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy() #(644.1024,3) is the same dimensional with the size of input test image in dataset.py. #201904101617tcw
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()#tcw201904101617
        bnd = scale  #tcw
        sr_1 = rgb2ycbcr(sr)
        hr_1 = rgb2ycbcr(hr)
        sr_1 = sr_1[bnd:-bnd,bnd:-bnd]#tcw201904111837
        hr_1 = hr_1[bnd:-bnd,bnd:-bnd]#tcw201904111837
        mean_psnr2 +=psnr(sr_1,hr_1)/len(dataset)
        mean_ssim += calculate_ssim(sr_1,hr_1)/len(dataset)
        print (name, psnr(sr_1,hr_1), calculate_ssim(sr_1,hr_1))
    print("PSNR promedio "+str(mean_psnr2))
    print("SSIM promedio " +str(mean_ssim))


def obtener_HR(path_LR, directorio):
    id_image = os.path.splitext(os.path.basename(path_LR))[0][:-2]
    if directorio == "Image Pairs":
        image_gt = os.path.join("testsets", "Test_Image_Pairs_CNN", "x2", id_image + "HR.png")
    else:
        image_gt = os.path.join("testsets", "Test_Flickr_CNN", "x2", id_image + "HR.png")
    return image_gt


def main(proceso, directorio, LR_path):
    module = importlib.import_module("model.{}".format("esrgcnn"))
    
    net = module.Net(multi_scale=True, group= 1)
    #print(json.dumps(vars(cfg), indent=4, sort_keys=True)) #print cfg information according order.
    state_dict = torch.load(os.path.join("Models","esrgcnn.pth"), map_location ='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") #0 is number of gpu, if this gpu1 is work, you can set it into 1 (device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
    net = net.to(device)
    if directorio == "Image Pairs":
        dataset = TestDataset(os.path.join("testsets","Test_Image_Pairs_CNN"), "2")
    else:
        dataset = TestDataset(os.path.join("testsets","Test_Flickr_CNN"), "2")

    if proceso == "test":
        sample(net, device, dataset)
    else:
        HR_path = obtener_HR(LR_path, directorio)
        hr_2 = Image.open(HR_path)
        lr_2 = Image.open(LR_path)
        hr = hr_2.convert("RGB")
        lr = lr_2.convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        hr = transform(hr)
        lr = transform(lr)
        t1 = time.time()
        lr = lr.unsqueeze(0).to(device)
        sr = net(lr, 2).detach().squeeze(0) #detach() break the reversed transformation.
        lr = lr.squeeze(0)
        t2 = time.time()
        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy() #(644.1024,3) is the same dimensional with the size of input test image in dataset.py. #201904101617tcw
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()#tcw201904101617
        bnd = 2  #tcw
        sr_1 = rgb2ycbcr(sr)
        hr_1 = rgb2ycbcr(hr)
        sr_1 = sr_1[bnd:-bnd,bnd:-bnd]#tcw201904111837
        hr_1 = hr_1[bnd:-bnd,bnd:-bnd]#tcw201904111837
        print("PSNR: ",psnr(sr_1,hr_1))
        print("SSIM: ",calculate_ssim(sr_1,hr_1))
        plt.figure()
        plt.subplot(131)
        plt.imshow(lr_2)
        plt.axis("off")
        plt.title("LR (256 x 256)")
        plt.subplot(132)
        plt.imshow(hr)
        plt.axis("off")
        plt.title("HR Ground Truth  (512 x 512)")
        plt.subplot(133)
        plt.imshow(sr)
        plt.axis("off")
        plt.title("HR Predicted (512 x 512)")
        plt.show()

        
"""
if __name__ == "__main__":
    main("test", "Flickr", os.path.join("testsets", "Test_Flickr_CNN", "x2", "002579_SRF_2_LR.png"))"""
