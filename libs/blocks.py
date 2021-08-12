import torch
import torch.nn as nn

from libs.layers import *
from libs.utils import num_flat_features

class GDefaultBlocks(nn.Module):
    # Default Blocks is PGGAN's Generator first blocks
    # Input Noise Vector Size = [Batch, LatentDim(defualt=512), 1, 1]
    # Output Image            = [Batch, Scale0_Feature, Scale0_H, Scale0_W]
    def __init__(self, in_channels, out_channels, relu=0.1, Scale0_H=4, Scale0_W=4):
        super(GDefaultBlocks, self).__init__()

        self.Scale0_H     = Scale0_H
        self.Scale0_W     = Scale0_W
        self.out_channels = out_channels

        self.pixelnorm  = PixelNormLayer(epslion=1e-8)
        self.dense      = nn.Linear(in_features= in_channels, out_features=Scale0_H*Scale0_W*out_channels)
        self.lrelu1     = nn.LeakyReLU(relu)
        self.pixelnorm2 = PixelNormLayer(epslion=1e-8)
        self.block_0    = CLP2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, x):
        # Defualt X shape : [Batch, Latent Dim Size, default_size_H, default_size_W]
        h = self.pixelnorm(x)

        # Make x shape : [Batch,  default_size_H * default_size_W * Latent Dim Size]
        h = h.view(-1, num_flat_features(h))

        # Make x shape : [Batch, Scale_0_H * Scale_1_W * Scale_0_Feature Size] 
        h = self.dense(h)
        h = self.lrelu1(h)
 
        # Make x shape : [Batch, Scale_0_Feature Size, Scale_0_H, Scale_1_W] 
        h = h.view(-1, self.out_channels, self.Scale0_H, self.Scale0_H)
        h = self.pixelnorm2(h)

        # Forwarding CLP Block for Scale 0
        h = self.block_0(h)

        return  h

class DDefaultBlocks(nn.Module):
    # Default Blocks is PGGAN's Discriminator first blocks
    # This block's poisition is always Last Layer


    def __init__(self, in_channels, out_channels, decision, kernel_size=3, stride=1, padding=1, relu=0.2, bias=True, Scale0_H=4, Scale0_W=4, MiniBatchstd=True):
        super(DDefaultBlocks, self).__init__()

        self.MiniBatchstd = MiniBatchstd
        self.minBatchstd = MiniBatchStandardDeviationLayer()
        self.block = EqulizedConv2DLayer(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.lrelu1 = nn.LeakyReLU(relu)
        self.linear = EqulizedLinearLayer(out_channels * Scale0_W * Scale0_H, out_channels)
        self.lrelu2 = nn.LeakyReLU(relu)
        self.decisionLayer = EqulizedLinearLayer(out_channels, decision)
    
    def forward(self, x):
        # So inputs Shape: [B, Scale1_Features, 4, 4]
        # if minibatchstd True
        if self.MiniBatchstd:
            x = self.minBatchstd(x)

        h = self.block(x)
        h = self.lrelu1(h)

        # Input Shape : [B, Scale0_Features, 4, 4]
        # Make Shape  : [B, 4 * 4 * Scale0_Features]
        h = h.view(-1, num_flat_features(h))

        # Shape : [B, Scale0_Features]
        h = self.linear(h)
        h = self.lrelu2(h)

        # Shape : [B, Decision(default: 1)]
        h = self.decisionLayer(h)
        return h



class GeneratorBlocks(nn.Module):
    # Generator Blocks For Up Scaling
    # Upsampling Nearest Neighbor Method
    # Forwarding 2 CLP Blocks

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, relu=0.1, bias=True, norm="pnorm"):
        super(GeneratorBlocks, self).__init__()

        blocks = []
        blocks += [nn.UpsamplingNearest2d(scale_factor=2)]
        blocks += [CLP2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=relu, bias=bias)]
        blocks += [CLP2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, relu=relu, bias=bias)]

        self.block = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.block(x)

class DiscriminaotrBlocks(nn.Module):
    # Discriminator Blocks For Down Scaling
    # 

    def __init__(self, in_channels, out_channels, pool_kernel=2, pool_stride=2,conv_kernel=3, conv_stride=1, conv_padding=1, conv_bias=True, relu=0.2):
        super(DiscriminaotrBlocks, self).__init__()
        blocks  = []
        blocks += [EqulizedConv2DLayer(in_channels, in_channels, conv_kernel, conv_stride, conv_padding, conv_bias)]
        blocks += [nn.LeakyReLU(relu)]
        blocks += [EqulizedConv2DLayer(in_channels, out_channels, conv_kernel, conv_stride, conv_padding, conv_bias)]
        blocks += [nn.LeakyReLU(relu)]
        blocks += [nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)]

        self.block = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.block(x)
        

if __name__ == "__main__":
    # Testing Generator's Default Block
    test_input = torch.randn((8, 512, 1, 1))
    block      = GDefaultBlocks(512, 512)
    test_out   =  block(test_input)
    print(f"Testing Generator's Default Block Done... : {test_out.shape}")

    # Testing Generator's Up-Scaling Blocks
    test_input = torch.randn((8, 512, 4, 4))
    block      = GeneratorBlocks(512, 512, 3, 1, 1)
    test_out   =  block(test_input)
    print(f"Testing Generator's Up-Scaling Block Done... : {test_out.shape}")

    # Testing Discriminator's Default Block
    test_input = torch.randn((8, 512, 4, 4))
    block      = DDefaultBlocks(513, 512, 1)
    test_out   = block(test_input)
    print(f"Testing Discriminator's Default Block Done... :  {test_out.shape}")

    # Testing Discriminator's Down-ScalingBlocks
    test_input = torch.randn((8, 256, 8, 8))
    block      = DiscriminaotrBlocks(256,512)
    test_out   = block(test_input)
    print(f"Testing Discriminator's Down-Scaling Block Done... :  {test_out.shape}")