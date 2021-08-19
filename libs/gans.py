from libs.losses import WGANGPGradientPenalty
import torch
import torch.nn as nn

from libs.models import *
from libs.losses import WGANGPGradientPenalty as WGANGP

class ProgressiveGrowingGAN(nn.Module):
    def __init__(self,
                latentDims = 512,
                output_dims = 3,
                decision=1,
                generationActivation=None,
                batchSize=8,
                lr = 1e-4,
                lambdaGP=10.0,
                miniBatch=True):
        super(ProgressiveGrowingGAN, self).__init__()       
        
        # Set Main Model's
        self.netG = Generator(latentDims, batchSize, dim_output=output_dims, scale0=latentDims, GenerationActivation=generationActivation)
        self.netD = Discriminator(batch_size=batchSize, dim_input=output_dims, scale0=latentDims, decision=decision, MiniBatchStd=miniBatch)

        self.optimG = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netG.parameters()), lr= lr, betas=[0, 0.99])
        self.optimD = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr= lr, betas=[0, 0.99])

        # Set Logs
        self.lossG = []
        self.lossD    = []
        self.lossWGANGP = []

        # Set Scalar
        self.alpha = 0
        self.lambdaGP = lambdaGP
        self.latentDims = latentDims
        self.batchSize = batchSize
    
    def getOutputSize(self):
        return self.netG.getOutputSize()
    
    def addScale(self, newScale):
        # Add Model's Scale
        self.netG.addSacle(newScale)
        self.netD.addSacle(newScale)
    
    def updateAlpha(self, newAlpha):
        # Update Alpha
        print(f"Chaning alpha to {newAlpha:.3f}")
        self.netG.setAlpha(newAlpha)
        self.netD.setAlpha(newAlpha)
        self.alpha = newAlpha
    
    def printInfo(self):
        # Just Print Information of GAN Model
        print(repr(self.netG))
        print(repr(self.netD))
    
    def oneStep(self, realInput):
        # Main Forward Pipeline
        # We just using WGAN-GP
        
        # Update the Discriminator
        self.optimD.zero_grad()

        # Bulid Noise Data
        fakeInput = torch.randn(self.batchSize, self.latentDims)

        # Real Part
        predReal = self.netD(realInput)
        loss_real = -torch.sum(predReal)

        # Fake Part
        predOutput = self.netG(fakeInput)
        predFake   = self.netD(predOutput)
        loss_fake  = torch.sum(predFake)

        # WGAN-GP Part
        self.lossWGANGP = [WGANGPGradientPenalty(realInput, predOutput, self.netD, self.lambdaGP)]

        # Loss Backward part
        loss_D = loss_real + loss_fake
        self.lossD += [loss_D.item()]
        loss_D.backward(retain_graph=True)

        # Update One Step
        self.optimD.step()

        # Update the Generator
        self.optimG.zero_grad()
        self.optimD.zero_grad()
        
        # Image Generation
        fakeInput = torch.randn(self.batchSize, self.latentDims)
        predOutput = self.netG(fakeInput)

        # Evaluation
        predFake = self.netD(predOutput)
        loss_G = torch.sum(predFake)
        self.lossG += [loss_G.item()]

        # Loss Backward
        loss_G.backward(retain_graph=True)

        # Update One Step
        self.optimG.step()
        






if __name__ == "__main__":
    # Testing Assert PGGAN
    pggan = ProgressiveGrowingGAN(latentDims=512)
    pggan.printInfo()
    pggan.addScale(256)
    pggan.printInfo()
    pggan.updateAlpha(1)
    pggan.printInfo()