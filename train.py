 
import torch 
import argparse
from utils.set_seed import set_seed

def main(cfg):
    algor=cfg.ALGORITHM.NAME
    if algor=='baseline':
        main_baseline(cfg)
    elif algor=='MixMatch':
        main_mixmatch(cfg)
    elif algor=='MTCF':
        main_mtcf(cfg)
    elif algor=='Ours':
        main_ours(cfg) 
    elif algor=='DASO':
        main_daso(cfg)    
    return 

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
 
    IF=[50,100]
    ood_r=[ 0.5]
    for if_ in IF:  # if
        # 同分布
        for r in ood_r: # r
            # if if_==10 and r==0.75 or if_==50 and r==0.5 or if_==100 and r in [0.5,0.75]:continue
            # if if_==10 and r==0:continue
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.freeze()
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            main(cfg)            
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
    
   