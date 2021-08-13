import torch
import torch.nn as nn

import copy

from libs.blocks import *
from libs.layers import EqulizedConv2DLayer

class Generator(nn.Module):
    def __init__(self, in_channels, batch_size=None, dim_output=3, scale0 = 512, relu=0.2, GenerationActivation=None):
        super(Generator, self).__init__()

        # Set Scalar of Generator
        self.image_default_H = 4
        self.image_default_W = 4
        self.alpha = 0
        self.scale = 0
        self.NowScale = 0
        self.relu = relu
        self.dim_output = dim_output
        self.in_channels = in_channels
        # self.scale_features = scale_features.copy()
        self.scale_features = [scale0]
        self.batch_size = batch_size

        # Set Module List of Generator
        self.Scale_Layer = nn.ModuleList() 
        self.toRGB_Layer = nn.ModuleList()

        # Set Default Block
        self.Scale_Layer.append(GDefaultBlocks(in_channels=in_channels, out_channels=scale0, relu=relu, Scale0_H=self.image_default_H, Scale0_W=self.image_default_W))
        self.toRGB_Layer.append(EqulizedConv2DLayer(scale0, dim_output, 1, 1, 0))

        # Set Additional Layer
        self.Upscale2d = nn.UpsamplingNearest2d(scale_factor=2)
        self.GenerationActivation = GenerationActivation

    def getOutputSize(self):
        # Find Generator's Now Output Size
        now_scale =  2 ** (len(self.toRGB_Layer) - 1)
        H = self.image_default_H * now_scale
        W = self.image_default_W * now_scale
        return (H, W)
    
    def addSacle(self, newScale):
        # Update Now Scale
        # Append New Layer

        LastSize = self.scale_features[- 1]
        self.scale_features.append(newScale)

        self.Scale_Layer.append(GeneratorBlocks(LastSize, newScale, 3, 1, 1, self.relu))
        self.toRGB_Layer.append(EqulizedConv2DLayer(newScale, self.dim_output, 1, 1, 0, True))
    
    def setAlpha(self, alpha):
        # Set New Alpha
        if alpha < 0 or alpha >1:
            raise ValueError("Alpha must be in [0, 1]")
        
        self.alpha = alpha
        
    def forward(self, x):

        # Forwarding Before Scale Layer
        for block in self.Scale_Layer[:-1]:
            x = block(x)

        # Alpha blending Image Generation
        if self.alpha > 0:
            y = self.toRGB_Layer[-2](x)
            y = self.Upscale2d(y)
        
        # Forwarding Last Scale Layer
        x = self.Scale_Layer[-1](x)

        # To RGB
        x = self.toRGB_Layer[-1](x)

        # Alpha Blending Alpha number moving
        if self.alpha >0:
            x = self.alpha * y + (1.0 - self.alpha) * x
        
        if self.GenerationActivation is not None:
            x = self.GenerationActivation(x)

        return x
    
    def __repr__(self):
        output_string = "==========================\n"
        output_string += "Generator's Information\n"
        output_string += "==========================\n"
        additional_string=""
        for index, features in enumerate(self.scale_features):
            now_scale =  (2 ** index)
            additional_string += f"block{index} : [{self.batch_size}, {self.image_default_H * now_scale}, {self.image_default_W * now_scale}, {features}]\n"
        return output_string+additional_string
        

        
    

class Discriminator(nn.Module):
    def __init__(self, batch_size=None,dim_input=3, scale0=512, relu=0.2, decision=1, MiniBatchStd=True):
        super(Discriminator, self).__init__()

        # Set Scalar of Discriminator
        self.image_default_H = 4
        self.image_default_W = 4
        self.alpha = 0
        self.scale = 0
        self.NowScale = 0
        self.dim_input = dim_input
        self.scale_features = [scale0]
        self.batch_size = batch_size
        self.decision = decision

        # Set Module List of Discriminator
        self.Scale_Layer = nn.ModuleList() 
        self.fromRGB_Layer = nn.ModuleList()

        self.fromRGB_Layer.append(EqulizedConv2DLayer(dim_input, scale0, 1, 1, 0))

        if MiniBatchStd:
            scale0 += 1

        self.Scale_Layer.append(DDefaultBlocks(scale0, scale0, decision, Scale0_W=self.image_default_W, Scale0_H=self.image_default_H, MiniBatchstd=MiniBatchStd))

        # Additional Layers
        self.MiniBatchStd = MiniBatchStd
        self.Pool2d  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lrelu   = nn.LeakyReLU(relu)
    
    def setAlpha(self, alpha):
        # Set New Alpha
        if alpha < 0 or alpha >1:
            raise ValueError("Alpha must be in [0, 1]")
        
        self.alpha = alpha

    def addSacle(self, newScale):
        # Update Now Scale
        # Append New Layer
        # In Scalar Layer Insert! Because, Discriminator is setting reversed Generator.

        self.NowScale += 1
        lastScale = self.scale_features[- 1]
        self.scale_features.append(newScale)

        self.Scale_Layer.insert(0, DiscriminaotrBlocks(newScale, lastScale))
        self.fromRGB_Layer.append(EqulizedConv2DLayer(self.dim_input, newScale, 1, 1, 0, True))
    
    def forward(self, x):
        # Inputs Shape [B, Scale Size, Scale Res H, Scale Res W]

        # For Alpha Blending 
        if self.alpha>0:
            y = self.Pool2d(x)
            y = self.fromRGB_Layer[-2](y)
            y = self.lrelu(y)
        
        # From RGB Layer
        x = self.lrelu(self.fromRGB_Layer[-1](x))

        # First Layer
        x = self.Scale_Layer[0](x)

        # If Alpha Blending on, Merge
        if self.alpha > 0:
            x = self.alpha * y + (1.0 - self.alpha) * x

        # Calculate Other Layer
        # If Scale 0, skip this step.
        for block in self.Scale_Layer[1:]:
            x = block(x)
        
        return x
        
    def __repr__(self):
        output_string = "==========================\n"
        output_string += "Discriminator's Information\n"
        output_string += "==========================\n"
        additional_string=""
        input_scale = 2 ** (self.NowScale)
        input_H = self.image_default_H * input_scale
        input_W = self.image_default_W * input_scale

        for index, features in enumerate(reversed(self.scale_features)):
            now_scale =  (2 ** index)
            additional_string += f"block{index} : [{self.batch_size}, {input_H // now_scale}, {input_W // now_scale}, {features}]\n"

        additional_string += f"\nDecision Block : [{self.batch_size}, {self.decision}]\n"
        additional_string += f"\nMini Batch std : {self.MiniBatchStd}" 
        return output_string + additional_string


if __name__ == "__main__":
    # Testing Scalar
    test_scale_features = [512, 512, 256, 256, 128, 128, 64]
    test_input = torch.randn(8, 512)

    # Testing Scale 0 
    # Testing Generator
    test_G = Generator(in_channels=512, scale0=test_scale_features[0], batch_size=8)
    test_output = test_G(test_input)
    print(repr(test_G))
    print(f"Generator's Result : {test_output.shape}")

    # Testing Discriminator
    test_D = Discriminator(scale0=test_scale_features[0], batch_size=8)
    test_fake = test_D(test_output)
    print(repr(test_D))
    print(f"\nDiscriminator's Result : {test_fake.shape}")

    # Testing Scale 1
    test_G.addSacle(test_scale_features[1])
    test_D.addSacle(test_scale_features[1])
    print("\n UPScaling...")

    # Testing Generator
    test_output = test_G(test_input)
    print(repr(test_G))
    print(f"Generator's Result : {test_output.shape}")

    # Testing Discriminator
    test_fake = test_D(test_output)
    print(repr(test_D))
    print(f"\nDiscriminator's Result : {test_fake.shape}")

    # Testing Set Alpha
    test_G.setAlpha(0.5)
    test_D.setAlpha(0.5)
    print("\n Set Alpha ... ")


    # Testing Generator
    test_output = test_G(test_input)
    print(repr(test_G))
    print(f"Generator's Result : {test_output.shape}")

    # Testing Discriminator
    test_fake = test_D(test_output)
    print(repr(test_D))
    print(f"\nDiscriminator's Result : {test_fake.shape}")