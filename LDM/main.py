import numpy as np
import sys
import ipywidgets as widgets
from LDM.notebook_helpers_yo import get_model, get_custom_cond, get_cond_options, get_cond, run
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch import linalg as LA
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm
import glob

class Initial_Dataset_Diffusion(Dataset):
    def __init__(self, LR_paths, HR_paths, transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])):
        self.HR = HR_paths
        self.LR = LR_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.LR)

    def __getitem__(self, index):
        x = self.LR[index]
        x = io.imread(x)
        x = self.transform(x)
        y = self.HR[index]
        y = io.imread(y)
        y = self.transform(y)
        return x, y, self.LR[index]


def calculate_psnr(img1, img2, crop_border):
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))



def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



def calculate_ssim(img1, img2, crop_border):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()



def modelo(image_LR, custom_steps=100, visualize=False):
    mode = widgets.Select(options=['superresolution'],value='superresolution', description='Task:')
    model = get_model(mode.value)
    logs = run(model["model"], image_LR, custom_steps, visualize=visualize)
    sample = logs["sample"]
    return sample


def final_run(directory ='Flickr_75_256_512', demo=False, path='002576'):
    mode = widgets.Select(options=['superresolution'],value='superresolution', description='Task:')
    model = get_model(mode.value)  
    dir, options = get_cond_options(mode.value)
    dir = directory
    if demo == False:
        cond_choice_path_LR = glob.glob(os.path.join(dir, 'LR', '*.png'))
        cond_choice_path_LR.sort()
        cond_choice_path_HR = glob.glob(os.path.join(dir, 'HR', '*.png'))
        cond_choice_path_HR.sort()
        metrics_diffusion(cond_choice_path_LR, cond_choice_path_HR, vis=False)
    else:
        cond_choice_path_LR = [os.path.join(dir, 'LR', path+'x2.png')]
        cond_choice_path_HR = [os.path.join(dir, 'HR', path+'.png')]
        metrics_diffusion(cond_choice_path_LR, cond_choice_path_HR, vis=True)


def metrics_diffusion(cond_choice_path_LR, cond_choice_path_HR, vis):
    batch_size = 1
    kwargs = {}
    custom_steps = 100
    dataset = Initial_Dataset_Diffusion(cond_choice_path_LR, cond_choice_path_HR)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size , shuffle=False, **kwargs)

    torch.manual_seed(20)
    psnrs = []
    ssims = []


    for LR, HR, path in tqdm(loader):
        sample = modelo(LR.to(torch.device("cuda")), visualize=vis)
        SR = sample
        SR = torch.clamp(SR, -1, 1).detach().cpu()
        SR = ((SR + 1.)/2. * 255)
        HR = torch.clamp(HR, 0, 1).detach().cpu()
        HR = (HR * 255)

        SR = np.transpose(SR.numpy(), (0, 2, 3, 1))
        HR = np.transpose(HR.numpy(), (0, 2, 3, 1))
        
        psnr = calculate_psnr(SR, HR, crop_border=0)
        ssim = calculate_ssim(SR[0], HR[0], crop_border=0)
        psnrs.append(psnr)
        ssims.append(ssim)

    psnrs = np.array(psnrs)
    ssims = np.array(ssims)

    print('\n','-'*100,'\n')
    pr = '' if vis == True else 'promedio'
    print('PSNR',pr,':', np.mean(psnrs))
    print('SSIM',pr,':', np.mean(ssims))

#-----------------------------------------------------------------------------------------------

 

