import cv2

import numpy as np

class Resize(object):

    # ReSizing Data

    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):

        for key, value in data.items():
            data[key] = cv2.resize(value, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)
        
        return data
        

class RandomFlip(object):

    # Random Fliping

    def __init__(self, vertical = True, horizontal=True):
        self.vertical = vertical
        self.horizontal = horizontal

    def __call__(self, data):

        if self.horizontal and np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flipud(value)

        if self.vertical and np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.fliplr(value)

        return data

class Normalization(object):
    
    # Normalized Data

    def __init__(self, mean=0.5, std= 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):

        for key, value in data.items():
            data[key] = (value - self.mean) / self.std