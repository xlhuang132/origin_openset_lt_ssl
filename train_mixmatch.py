
from logging import getLoggerClass
from operator import mod
from tkinter import W
import torch 
import argparse
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np
from dataset.build_dataloader import build_dataloader
from loss.build_loss import build_loss 
import models 
import time 
import torch.optim as optim
import os   
import logging
from utils.set_seed import set_seed 
import torch.nn.functional as F
from utils import AverageMeter, accuracy, create_logger,\
                    plot_group_acc_over_epoch,prepare_output_path,\
                    interleave,plot_loss_over_epoch
from utils.build_optimizer import get_optimizer, get_scheduler
from utils.utils import linear_rampup
from utils.validate_model import validate
from trainer import MixMatchTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="codes for MixMatch")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/mixmatch_cifar10.yaml",
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
    
    IF=[100]
    ood_r=[0.75]
    for if_ in IF:   
        for r in ood_r:  
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.freeze()
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            # main_mixmatch(cfg)   
            trainer=MixMatchTrainer(cfg)
            trainer.train()     
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
   
    
    
    
    

