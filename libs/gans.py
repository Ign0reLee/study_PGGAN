import torch
import torch.nn as nn

from libs.models import *

class ProgressiveGrowingGAN(nn.Module):
    def __init__(self,
                latent_dims = 512,
                output_dims = 3,
                decision=1,
                generationActivation=None,
                scale_features=[512, 512, 256, 256, 64, 32],
                batch_size=8,
                miniBatch=True):
        super(ProgressiveGrowingGAN, self).__init__()

        assert scale_features[0] == latent_dims, f"Latent Dim's Size and Scale 0 Features Size must be same! Now : [latent dim : {latent_dims}, scale 0 : {scale_features[0]} ]"
        
        
        # Set Main Model's
        self.netG = Generator(latent_dims, batch_size, dim_output=output_dims, scale_features=scale_features, GenerationActivation=generationActivation)
        self.netD = Discriminator(batch_size=batch_size, dim_input=output_dims, scale_features=scale_features, decision=decision, MiniBatchStd=miniBatch)
        self.printInfo()

        # Set Scalar
        self.alpha = 0
    
    def getOutputSize(self):
        return self.netG.getOutputSize()
    
    def addScale(self):
        self.netG.addSacle()
        self.netD.addSacle()
        self.printInfo()
    
    def updateAlpha(self, newAlpha):
        print(f"Chaning alpha to {newAlpha:.3f}")
        self.netG.setAlpha(newAlpha)
        self.netD.setAlpha(newAlpha)
        self.alpha = newAlpha
    
    def printInfo(self):
        print(repr(self.netG))
        print(repr(self.netD))
        
        






if __name__ == "__main__":
    # Testing Assert PGGAN
    pggan = ProgressiveGrowingGAN()
    pggan.addScale()
    pggan.updateAlpha(1)