import torch
import torch.nn as nn

import numpy as np

class MiniBatchStandardDeviationLayer(nn.Module):
    # MiniBatchstd Layer
    def __init__(self, epslion=1e-8):
        super(MiniBatchStandardDeviationLayer, self).__init__()
        self.epslion = epslion
    
    def forward(self, x, subGroupSize=4):
        size = x.size()
        subGroupSize = min(size[0], subGroupSize)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1:
            # Divide inputs for minibatch calculating
            y = x.view(-1, subGroupSize, size[1], size[2], size[3])
            # Calc Variance of Sub Group's
            y = torch.var(y, 1)
            # Calc Standard Deviation For 
            y = torch.sqrt(y + self.epslion)
            # Reshape Group
            y = y.view(G, -1)
            # Calc Mean of y(axis=1) and Reshape Same Size
            y = torch.mean(y, 1).view(G, 1)
            # Tiled Same as Inputs
            y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
        else:
            # If Sub Group is 0, minibatchstd same
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        return torch.cat([x, y], dim=1)

    


class PixelNormLayer(nn.Module):
    # Pixel Normalization Layer
    # Calculate b_xy = a_xy/root(mean(a_xy)^2 + epslion)
    # Epslion : 1e-8(default)

    def __init__(self, epslion=1e-8):
        super(PixelNormLayer, self).__init__()
        self.epslion = epslion

    def forward(self, x):
        return  x * torch.rsqrt(torch.mean(x **2, dim=1, keepdim=True) + self.epslion)
    
    def __repr__(self):
        return self.__class__.__name__

class WScaleLayer(nn.Module):
    # Weight Scale Layer For Equlized Learning Rate
    # Module     : Layer(Like nn.Conv2d, nn.Linear)
    # lrMul      : 1.0(float)
    # gain       : 2.0(float)
    # biasToZero : True(bool)
    # equlized   : True(bool)

    def __init__(self, module, lrMul=1.0, gain=2.0, biasToZero=True, equlized=True):
        super(WScaleLayer, self).__init__()

        self.module = module
        self.gain = gain
        self.equilized = equlized
        self.biasToZero = biasToZero

        if self.biasToZero:
            # Make Module's Bias To Zero
            self.module.bias.data.fill_(0)

        if self.equilized:
            # Make Module Weight To (0,1) Normaliy
            self.module.weight.data.normal_(0,1)
            # Make Moduel Weight To (0, 1/lrMul), By Default lrMul is  1.0
            self.module.weight.data /= lrMul
            # Calculate Equlized Weight
            self.getLayerNormalizationFactor()

    def forward(self, input):
        x = self.module(input)
        if self.equilized:
            x = x * self.scale
        return x 
    
    def __repr__(self):
        return f"{self.__class__.__name__} (gain={self.gain})"
    
    def getLayerNormalizationFactor(self):

        # Find Now Module's Weight
        self.size = np.array(self.module.weight.size())

        # To Make [Feature Size, Filter Number, Kernel Size, Kernel Size]
        self.size[1], self.size[0] = self.size[0], self.size[1]

        # To Calculate He's Initializer Constant Value
        fan_in = np.prod(self.size[1:])
        self.scale = np.sqrt(self.gain / fan_in)
        

class EqulizedLinearLayer(WScaleLayer):
    # Make Linear Layer Equlized
    # in_channels  : (int)
    # out_channles : (int)

    def __init__(self, in_channels, out_channels):
        WScaleLayer.__init__(self, nn.Linear(in_channels, out_channels))

class EqulizedConv2DLayer(WScaleLayer):
    # Make Conv2d Layer Equlized
    # in_channels  : (int)
    # out_channles : (int)
    # kernel_size  : 3(int)
    # stride       : 1(int)
    # padding      : 1(int)

    def __init__(self, in_channels, out_channels, kernel_size=3 , stride=1, padding=1, bias=True):
        WScaleLayer.__init__(self, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    
    
class CLP2d(nn.Module):
    # Default Conv2d, Leaky ReLU, Pixel Normalization Block
    # Conv2d Always Equlized Conv2d Layer
    # Leaky ReLU can make ReLU see, relu parameter below
    # Now support just Pixel Normalization because this code just using PGGAN
    # in_channels  : (int)
    # out_channles : (int)
    # kernel_size  : (int)
    # stride       : (int)
    # padding      : (int)
    # relu         : 0.1(float)
    # bias         : True(bool)
    # norm         : pnorm(string)

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, relu=0.1, bias=True, norm="pnorm"):
        super(CLP2d, self).__init__()

        layers =[]
        layers += [EqulizedConv2DLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]
        
        if norm is "pnorm":
            layers +=[PixelNormLayer()]

        self.clp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.clp(x)



if __name__ == "__main__":
    # Testing Equlized Linear Layer
    test_input_linear = torch.randn(4)
    equlized_linear=EqulizedLinearLayer(4,3)
    out = equlized_linear(test_input_linear)
    print(f"Equlized Linear Done.. : {out.shape}")

    # Testing Equlized Conv Layer
    test_input_conv = torch.randn((1, 3, 9, 9))
    equlized_conv = EqulizedConv2DLayer(3, 512, 3, 1, 1)
    out = equlized_conv(test_input_conv)
    print(f"Equlized Conv2d Layer Done.. : {out.shape}")

    # Testing Equlized CLP2d Block
    equlized_convblock = CLP2d(3, 512, 3, 1, 1)
    out = equlized_convblock(test_input_conv)
    print(f"Equlized CLP2d Block Done.. : {out.shape}")