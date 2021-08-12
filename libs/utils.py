import os

import torch
import torch.nn as nn


def num_flat_features(x):
    # Find Number of all Feautre size without Batch size

    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def save(ckpt_dir, model, optim, epoch, step, model_name="PGGAN"):

    torch.save({"model": model.state_dict(), "optimizer" : optim.state_dict()}, os.path.join(ckpt_dir, model_name, f"{model_name}_{epoch}_{step}.pth"))

def load(ckpt_dir, model, optim, epoch=None, step=None):
    
    ckpt_lst = None

    if epoch is not None:
        ckpt_lst = [i for i in os.listdir(ckpt_dir) if int(i.split("_")[-2]) == epoch]
    
    print(ckpt_lst)
    if step is not None:
        if ckpt_lst is not None:
            ckpt_lst = [i for i in ckpt_lst if i.split("_")[-1][0] == step]
        else:
            ckpt_lst = [i for i in os.listdir(ckpt_dir) if int(i.split("_")[-1][:-4]) == step]
    
    print(ckpt_lst)
    
    if ckpt_lst is None:
        ckpt_lst = os.listdir(ckpt_dir)

    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


    # Load Epoch And Step
    epoch = int(ckpt_lst[-1].split("_")[-2])
    step = int(ckpt_lst[-1].split("_")[-1][:-4])
    print(epoch)
    print(step)
    print(ckpt_lst)

    # Load Model
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))
    model.load_state_dict(dict_model['model'])
    optim.load_state_dict(dict_model["optimizer"])
    

    return model, optim, epoch, step

if __name__ == "__main__":
    # Load Testing
    load(ckpt_dir="ckpt_dir", model=None, optim=None, epoch=2,step=None)
