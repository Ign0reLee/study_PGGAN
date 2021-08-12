import torch
import torch.nn

from libs.gans import ProgressiveGrowingGAN as PGGAN
from libs.losses import WGANGPGradientPenalty as WGANGP


class Trainer():
    def __init__(self):
        self.model = PGGAN()
    
    def updateAlphaJumps(self, nJumpScale, sizeJumpScale):
        
