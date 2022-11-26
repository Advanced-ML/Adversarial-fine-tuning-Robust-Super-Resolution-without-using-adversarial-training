import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm 
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Initial_Dataset(Dataset):
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






