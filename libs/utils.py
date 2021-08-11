import torch
import torch.nn as nn

def num_flat_features(x):
    # Find Number of all Feautre size without Batch size

    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features