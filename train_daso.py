
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np
import models 
import time 
import os   
from trainer import DASOTrainer
from utils.set_seed import set_seed
def parse_args():
    parser = argparse.ArgumentParser(description="codes for DASO")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/daso_cifar10.yaml",
        type=str,
    ) 
    
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    # set random seed
    set_seed(cfg.SEED)
    cudnn.benchmark = True  
    IF=[100] # 10,50,100
    
    ood_r=[0.5,0.75] # basline不用mixup的话不用考虑r 0.0,0.25, 0.5, 0.75,1.0 randomsampler+classreversedsampler没有用到mixup
 
    for if_ in IF:  # if
        # 同分布
        for r in ood_r: 
            # if r==0 and if_==10 :continue
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            cfg.freeze()      
            trainer=DASOTrainer(cfg)
            trainer.train()
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
    
    
    

