import os

import torch
import torch.nn as nn
import torch.distributed as dist


def num_flat_features(x):
    # Find Number of all Feautre size without Batch size

    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def save(ckpt_dir, netG, netD, optimG, optimD, scale, step, model_name="PGGAN"):
    r"""
    Model Saver

    Inputs:
        ckpt_dir   : (string) check point directory
        netG       : (nn.module) Generator Network
        netD       : (nn.module) Discriminator Network
        opitmG     : (torch.optim) Generator's Optimizers
        optimD     : (torch.optim) Discriminator's  Optimizers
        scale      : (int) Now Scale
        step       : (int) Now Step
        model_name : (string) Saving model file's name
    """
    if hasattr(netG, "module"):
        netG_dicts =  netG.module.state_dict()
        netD_dicts =  netD.module.state_dict()
        try:
            optimG_dicts = optimG.moudle.satae_dict()
            optimD_dicts = optimD.moudle.satae_dict()
        except:
            optimG_dicts = optimG.satae_dict()
            optimD_dicts = optimD.satae_dict()

    else:
        netG_dicts =  netG.state_dict()
        netD_dicts =  netD.state_dict()
        optimG_dicts = optimG.satae_dict()
        optimD_dicts = optimD.satae_dict()

    torch.save({"netG": netG_dicts,
                "netD": netD_dicts,
                "optimG" : optimG_dicts,
                "optimD" : optimD_dicts},
                os.path.join(ckpt_dir, model_name, f"{model_name}_{scale}_{step}.pth"))

def load(ckpt_dir, netG, netD, optimG, optimD, scale=None, step=None, gpu=None):
    r"""
    Model Lodaer

    Inputs:
        ckpt_dir : (string) check point directory
        netG     : (nn.module) Generator Network
        netD     : (nn.module) Discriminator Network
        opitmG   : (torch.optim) Generator's Optimizers
        optimD   : (torch.optim) Discriminator's  Optimizers
        scale    : (int) find scale. if None, last scale
        step     : (int) find step.  if NOne, last scale
    """
    
    ckpt_lst = None

    if scale is not None:
        ckpt_lst = [i for i in os.listdir(ckpt_dir) if int(i.split("_")[-2]) == scale]
    
    if step is not None:
        if ckpt_lst is not None:
            ckpt_lst = [i for i in ckpt_lst if i.split("_")[-1][0] == step]
        else:
            ckpt_lst = [i for i in os.listdir(ckpt_dir) if int(i.split("_")[-1][:-4]) == step]
    
    if ckpt_lst is None:
        ckpt_lst = os.listdir(ckpt_dir)

    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


    # Load Scale And Step
    scale = int(ckpt_lst[-1].split("_")[-2])
    step = int(ckpt_lst[-1].split("_")[-1][:-4])

    # Load Model
    if gpu is not None:
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]), map_location=mapLocation)
    else:
        dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))

    try:
        netG.load_state_dict(dict_model['netG'])
        netD.load_state_dict(dict_model['netD'])
    except:
        netG.module.load_state_dict(dict_model['netG'])
        netD.module.load_state_dict(dict_model['netD'])
        
    optimG.load_state_dict(dict_model["optimG"])
    optimD.load_state_dict(dict_model["optimD"])
    
    return netG, netD, optimG, optimD, scale, step

def init_process(rank, size, path, backend="nccl"):
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_IB_DISABLE']= '1'
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(backend=backend, init_method='file://'+path+'/sharedfile', rank=rank, world_size=size)

if __name__ == "__main__":
    # Load Testing
    load(ckpt_dir="ckpt_dir", model=None, optim=None, epoch=2,step=None)
