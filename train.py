 
import torch 
import argparse
from utils.set_seed import set_seed
from trainer.build_trainer import build_trainer
from config.defaults import update_config,_C as cfg
import torch.backends.cudnn as cudnn   

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
    ood_r=[0.75]
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
            # main(cfg)     
            trainer=build_trainer(cfg)     
            trainer.train()  
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
    
   