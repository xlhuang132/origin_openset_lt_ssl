
from logging import getLoggerClass
from operator import mod
from tkinter import W
import torch 
import logging
import argparse
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np
from dataset.build_dataloader import build_dataloader
from dataset.build_sampler import build_sampler
from loss.build_loss import build_loss 
import models 
import time 
import torch.optim as optim
import os   
from utils.set_seed import set_seed
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import accuracy,AverageMeter,FusionMatrix,plot_acc_over_epoch,plot_accs_zhexian,\
    plot_group_acc_over_epoch,plot_loss_over_epoch,create_logger,prepare_output_path,\
        get_warmup_model_path,load_checkpoint
from utils.build_optimizer import get_optimizer, get_scheduler
from utils.utils import linear_rampup 
from loss.get_class_weight import get_class_weight
from utils.validate_model import validate
import math
from tqdm import tqdm
from utils.mixup import mix_up
from models.ood_detector import ood_detect_batch 

from trainer import ACRTrainer
    
def parse_args():
    parser = argparse.ArgumentParser(description="codes for Moving Center")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/moving_center_cifar10.yaml",
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
    # gpu0: # OOD_dataset["TIN","LSUN","Gau","Uni"]
    #    
    #   cifar100:  IF=[10,50,100] ood_r=[0.5] "TIN"
    
    # gpu1:
    #   CIFAR10:  IF=[50,100] ood_r=[ 0.5 ] "TIN"
    #       
    #     
 
    IF=[100]
    ood_r=[0.5]
    for if_ in IF:  # if
        # 同分布
        for r in ood_r: # r 
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.freeze()
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            # load_warmup_model(None, cfg)
            # main_ours(cfg) 
            trainer=ACRTrainer(cfg)
            trainer.train()  
            # trainer.evaluate()         
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
    
    
    
    

