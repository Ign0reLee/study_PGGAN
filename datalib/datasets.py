import os
import cv2
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.lst_data = os.listdir(self.data_dir)
        self.lst_data.sort()

        self.to_tensor = ToTensor()
    
    def __len__(self):
        return len(self.lst_data)
    
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_dir, self.lst_data[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     

        # Image Scaling
        if img.dtype == np.uint8:
            img = img / 255.0

        data = {"real_img" :img}

        # Data Transforming
        if self.transform:
            data = self.transform(data)
        
        data = self.to_tensor(data)

        return data

class ToTensor(object):

    # Make Image [B, H, W, C] to [B, C, H, W]
    # Make Image Numpy Array to Tensor

    def __call__(self, data):

        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

