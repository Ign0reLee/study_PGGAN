import torch
import torch.nn as nn

from .blocks import *
from .layers import EqulizedConv2DLayer

class Generator(nn.Module):
    def __init__(self, in_channels, dim_output=3, scale_features=[512, 512, 256, 256], relu=0.2, block=5, size=1, GenerationActivation=None):
        super(Generator, self).__init__()

        # Set Scalar of Generator
        self.image_default_H = 4
        self.image_default_W = 4
        self.alpha = 0
        self.scale = 0
        self.NowScale = 0
        self.relu = relu
        self.size = size
        self.dim_output = dim_output
        self.in_channels = in_channels
        self.scale_features = scale_features

        # Set Module List of Generator
        self.Scale_Layer = nn.ModuleList() 
        self.toRGB_Layer = nn.ModuleList()

        # Set Default Block
        self.Scale_Layer.append(GDefaultBlocks(in_channels=in_channels, dense_channels=scale_features[self.NowScale], relu=relu, Scale0_H=self.image_default_H, Scale0_W=self.image_default_W))
        self.toRGB_Layer.append(EqulizedConv2DLayer(scale_features[self.NowScale], dim_output, 1, 1, 0))

        # Set Additional Layer
        self.Upscale2d = nn.UpsamplingNearest2d(scale_factor=2)
        self.GenerationActivation = GenerationActivation

    def getOutputSize(self):
        # Find Generator's Now Output Size
        now_scale =  (2 ** len(self.toRGB_Layer) - 1)
        H = self.image_default_H * now_scale
        W = self.image_default_W * now_scale
        return (H, W)
    
    def addSacle(self):
        # Update Now Scale
        # Append New Layer

        self.NowScale += 1
        LastSize = self.scale_features[self.NowScale - 1]
        NewSize  = self.scale_features[self.NowScale]

        self.Scale_Layer.append(GeneratorBlocks(LastSize, NewSize, 3, 1, 1, self.relu))
        self.toRGB_Layer.append(EqulizedConv2DLayer(NewSize, self.dim_output, 1, 1, 0, True))
    
    def SetAlpha(self, alpha):
        # Set New Alpha
        if alpha < 0 or alpha >1:
            raise ValueError("Alpha must be in [0, 1]")
        
        self.alpha = alpha
        
    def forward(self, x):
        # For First step
        scale = None

        # Forwarding Before Scale Layer
        for scale, block in enumerate(self.Scale_Layer[:-1]):
            # if scale is not 0, it means need to be upscaling
            if scale is not 0:
                x = self.Upscale2d(x)
            x = block(x)

        # Alpha blending Image Generation
        if self.alpha > 0:
            y = self.toRGB_Layer[-2](x)
            y = self.Upscale2d(y)
        
        # Forwarding Last Scale Layer
        if scale is not None:
            x = self.Upscale2d(x) # need to be upscaling
        x = self.Scale_Layer[-1](x)

        # To RGB
        x = self.toRGB_Layer[-1](x)

        # Alpha Blending Alpha number moving
        if self.alpha >0:
            x = self.alpha * y + (1.0 - self.alpha) * x
        
        if self.GenerationActivation is not None:
            x = self.GenerationActivation(x)

        return x
        

        
    

class Discriminator(nn.Module):
    
    def __init__(self, in_channels, dim_output=3, scale_features=[512, 512, 256, 256], relu=0.2, block=5, size=1, GenerationActivation=None):
        super(Discriminator, self).__init__()

        # Set Scalar of Generator
        self.image_default_H = 4
        self.image_default_W = 4
        self.alpha = 0
        self.scale = 0
        self.size = size
        self.dim_output = dim_output
        self.in_channels = in_channels
        self.scale_features = scale_features

        # Set Module List of Generator
        self.Scale_Layer = nn.ModuleList() 
        self.toRGB_Layer = nn.ModuleList()


if __name__ == "__main__":
    test_G = Generator(in_channels=64, dense_channels=1024, relu=0.1, block=5)