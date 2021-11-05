import gc, os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torchvision import transforms
from torchvision import utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP

from datalib.datasets import Dataset
from datalib.compose import Normalization, RandomFlip, Resize

from libs.utils import init_process, save, load
from libs.gans import ProgressiveGrowingGAN as PGGAN





class Trainer():
    def __init__(self,
                path,
                batchSize=8,
                ckptDir="ckpt_dir",
                modelName = "FFHQ_PGGAN",
                latentDims=512,
                decision=1,
                restart=False,
                saveIter=12000,
                nJumpScale=600,
                sizeJumpScale=32,
                nIterations=[24000, 24000, 24000, 24000, 24000],
                scale_features=[512, 512, 256, 256, 128],
                numWorker=os.cpu_count()//2):
        
        assert scale_features[0] == latentDims, f"Latent Dim and Scale Features 0 must be same value! Now : [Latent Dim: {latentDims}, Scale 0: {scale_features[0]}]"
        assert len(nIterations) == len(scale_features), f"Iteraion's Info must be same as Scale Features! Now : [Iterations: {len(nIterations)}, ScaleFeatures: {len(scale_features)}]"

        self.model = PGGAN(latentDims=latentDims, decision=decision, batchSize=batchSize)

        self.nIterations = nIterations
        self.scale_features = scale_features
        self.sizeJumpScale = sizeJumpScale
        self.nJumpScale = nJumpScale
        self.saveIter = saveIter
        self.restart = restart
        self.ckptDir = ckptDir
        self.modelName = modelName
        self.path = path
        self.numWorker = numWorker
        self.batchSize = batchSize
        self.nowH, self.nowW = self.model.getOutputSize()
        self.sharedFilePath = os.path.dirname(os.path.abspath(__file__))

        self.startScale = 0
        self.startStep  = 0

        # # Make Default DataSet
        self.transform_train = transforms.Compose([Resize((self.nowH, self.nowW)), RandomFlip(horizontal=False), Normalization(mean=0.5, std=0.5)])
        self.dataset_train = Dataset(data_dir=path, transforms=self.transform_train)
        self.dataset_sampler = DistributedSampler(self.dataset_train)
        self.loader_train = DataLoader(self.dataset_train, batch_size=self.batchSize, shuffle=False, num_workers=self.numWorker,
                                        pin_memory=True, sampler=self.dataset_sampler)

        # Output Additional Function
        self.fn_toNumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 3, 1)
        self.fn_toDenorm = lambda x, mean, std: (x * std) + mean
    
    def updateAlphaJumps(self, alpha):

        diffJump = 1.0 / float(self.nJumpScale)
        return alpha - diffJump

    
    def train(self, gpu, size):

        if self.restart:
            print("Model Loading...")
            self.model.netG, self.model.netD, self.model.optimG, self.model.optimD, self.startScale, self.startStep = load(self.model.netG, self.model.netD, self.model.optimG, self.model.optimD)
            print(f"Done... Scale : {self.startScale}, Step : {self.startStep}")
        
        torch.cuda.set_device(gpu)
        init_process(gpu, size, self.sharedFilePath)

        self.model.cuda(gpu)
        self.model = DDP(self.model, device_ids=[gpu])

        for index, scale in enumerate(self.scale_features):
            # Model Add Scale and Print Information
            # If index not 0(if not scale 0), set Alpha 1 and add Modle's Scale
            if index is not 0:
                # Update Model
                self.model.addScale(scale)
                self.model.alpha = 1.0
                self.nowH, self.nowW = self.model.getOutputSize()

                # Make Loader New Scale
                self.transform_train = transforms.Compose([Resize(self.nowH, self.nowW), RandomFlip(horizontal=False), Normalization(mean=0.5, std=0.5)])
                self.dataset_train = Dataset(data_dir=self.path, transforms=self.transform_train)
                self.dataset_sampler = DistributedSampler(self.dataset_train)
                self.loader_train = DataLoader(self.dataset_train, batch_size=self.batchSize, shuffle=False, num_workers=self.numWorker,
                                                pin_memory=True, sampler=self.dataset_sampler)

                gc.collect()
            
            if self.startScale > index:
                # If restart on.
                # Load Start Scale
                continue

            print(f"\nScale {index}")
            print(f"Size : {self.model.getOutputSize()} \n")
            self.model.printInfo()

            # Main Iteration
            for nowIter in range(self.startStep, self.nIterations[index]):

                # If Iterationg JumpScale, Update Alpha
                if self.model.alpha > 0 and nowIter == self.nJumpScale:
                     alpha = self.updateAlphaJumps(alpha=self.model.alpha) 
                     self.model.updateAlpha(alpha)

                # Main Iteration
                for data in self.loader_train:
                    real_input = data["real_img"].cuda(gpu, non_blocking=True)
                    self.model.oneStep(real_input)

                # Print 100 Iteration
                if nowIter % 100 == 0 or self.nIterations[index] == nowIter:
                    print(f"Scale {index} | Iter {nowIter} / {self.nIterations[index]} | Loss G {np.mean(self.model.lossG)} | Loss D {np.mean(self.model.lossD)}")
                    self.model.lossG      = []
                    self.model.lossD      = []
                    self.model.lossWGANGP = []

                # If iterating save iter, model save
                if nowIter % self.saveIter == 0:
                    self.checkPoint(gpu,index, nowIter)
                    img = self.outputSave()
                    plt.imsave(os.path.join(self.ckptDir, self.modelName, f"{self.modelName}_{index}_{nowIter}.png"), img)

            # Initialize Some Setteing
            if self.startStep > 0:
                self.startStep = 0
    
    def outputSave(self):
        output = self.model.makeImage()
        image = np.clip(self.fn_toNumpy(self.fn_toDenorm(output, 0.5, 0.5)).squeeze(), a_min=0, a_max=1)
        batch, _,_,_ = np.shape(image)
        
        return np.concatenate((np.concatenate(image[:batch//2], axis=1), np.concatenate(image[batch//2:], axis=1)),axis=0)
        
                
    def checkPoint(self, gpu, index, nowIter):

        if gpu == 0:
            save(self.ckptDir, self.model.netG, self.model.netD, self.model.optimG, self.model.optimD, index, nowIter, self.modelName)
        
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(os.path.join(self.ckptDir, self.modelName, f"{self.modelName}_{index}_{nowIter}.pth"), map_location=mapLocation)
        self.model.netG.module.load_state_dict(dict_model["netG"])
        self.model.netD.module.load_state_dict(dict_model["netD"])

    def runTrain(self):

        gpus = torch.cuda.device_count()
        mp.spawn(self.train, args=(gpus,), nprocs=gpus)


            
   
    

if __name__ == "__main__":
    trainer = Trainer("../Data/")
    trainer.train(None)

        
