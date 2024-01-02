
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
import logging
import time 
import torch.optim as optim
import os   
import torch.nn.functional as F
from utils import AverageMeter, accuracy, create_logger,\
    plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch,plot_acc_over_epoch
from utils.build_optimizer import get_optimizer, get_scheduler
from utils.utils import linear_rampup
from utils.validate_model import validate
from utils.set_seed import set_seed
from loss.get_class_weight import get_class_weight
from utils import FusionMatrix
from utils.ema_model import EMAModel

from trainer import FixMatchTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/fixmatch_cifar10.yaml",
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
    ood_r=[0.5]
    for if_ in IF:   
        for r in ood_r:  
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.freeze()
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            # main_fixmatch(cfg) 
            trainer=FixMatchTrainer(cfg)
            trainer.train()           
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
     
   
    
    
    
    

