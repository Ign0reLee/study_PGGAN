import os, glob, json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from datalib import *
from libs.trainer import *

# Make Parser
parser = argparse.ArgumentParser(description="Face Generate From Small Landmark Point",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--mode",  default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument('-c', '--config', help="Model's name", type=str, dest="config")
parser.add_argument("-n", "--name", help="Model's Name", type=str, dest="name")
parser.add_argument("-o", "--outPath", default="resultModel",help="Output Directory", type=str, dest="outPath")
parser.add_argument("-s", "--saveIter", help="Save Iteration", type=int, dest="saveIter")
parser.add_argument("--restart", help="If True, Restart All", action="store_true")

# Set Parameter
args = parser.parse_args()
mode = args.mode
config = args.config
nameModel = args.name
outPath = args.outPath
saveIter = args.saveIter
restart = args.restart


# If Output Path Directory is None, Make Directory
if not os.path.exists(outPath):
    os.mkdir(outPath)

# Model's Name Output
outPath = os.path.join(outPath, nameModel)
if not os.path.exists(outPath):
    os.mkdir(outPath)


trainer = Trainer()


if __name__ == "__main__":

    assert nameModel is not None, f"Please Input Model's Name"

    # Print String
    print("====================")
    print(f"Mode: {mode}")
    if restart:
        print(f"Restart : {restart}")
    print(f"Model Name : {nameModel}")
    print("====================")