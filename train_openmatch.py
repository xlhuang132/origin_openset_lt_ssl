
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
    plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch
from utils.build_optimizer import get_optimizer, get_scheduler
from utils.utils import linear_rampup
from utils.validate_model import validate
from utils.set_seed import set_seed 
from loss.open_match_loss import ova_loss
from trainer import OpenMatchTrainer
def parse_args():
    parser = argparse.ArgumentParser(description="codes for openmatch")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/cifar10_openmatch.yaml",
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
    

def train(labeled_trainloader, unlabeled_trainloader, \
            model, optimizer, l_criterion,ul_criterion, \
            epoch,logger,cfg,
            dual_labeled_trainloader=None):
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter() 
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()
    total_step=cfg.TRAIN_STEP
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    dual_labeled_train_iter=None
    if dual_labeled_trainloader!=None:
        dual_labeled_train_iter=iter(dual_labeled_trainloader)
    # ====== something for open match =======
    losses_o = AverageMeter()
    losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    start_fix=cfg.ALGORITHM.OPEN_SAMPLING.START_FIX
    threshold=cfg.ALGORITHM.CONFIDENCE_THRESHOLD
    T=cfg.ALGORITHM.OPEN_SAMPLING.T
    
    lambda_oem=cfg.ALGORITHM.OPEN_SAMPLING.LAMBDA_OEM
    lambda_socr=cfg.ALGORITHM.OPEN_SAMPLING.LAMBDA_SOCR
    amp=cfg.ALGORITHM.OPEN_SAMPLING.AMP
    # ========================================
    for i in range(1,total_step+1):
        loss =0
        # DL 数据
        try:
            inputs_x, targets_x = labeled_train_iter.next() 
        except:
            labeled_train_iter = iter(labeled_trainloader)          
            inputs_x, targets_x = labeled_train_iter.next()
        dual_inputs_x=None
        if dual_labeled_train_iter!=None:
            try: 
                dual_inputs_x, dual_targets_x=dual_labeled_train_iter.next()
            except:        
                dual_labeled_train_iter=iter(dual_labeled_trainloader) 
                dual_inputs_x, dual_targets_x=dual_labeled_train_iter.next()
        
        # DU 数据
        try:
            (inputs_u, inputs_u2), _, = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _, = unlabeled_train_iter.next()
        # 限制两批数量相等
        bs=inputs_x.size(0)
        if bs<inputs_u.size(0):
            inputs_u,inputs_u2=inputs_u[:bs],inputs_u2[:bs]

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        
        # 是否进行mixup
        if cfg.ALGORITHM.BRANCH1_MIXUP:
            inputs_x=mix_up(inputs_x, inputs_u)
        if dual_inputs_x!=None:
            dual_inputs_x, dual_targets_x = dual_inputs_x.cuda(), dual_targets_x.cuda(non_blocking=True)
            if cfg.ALGORITHM.BRANCH2_MIXUP:
                dual_inputs_x=mix_up(dual_inputs_x, inputs_u)
            inputs_x=torch.cat((inputs_x,dual_inputs_x),dim=0)
            targets_x=torch.cat((targets_x,dual_targets_x),dim=0)
        #   ==================================
        try:
            (_, inputs_x_s, inputs_x), targets_x = labeled_train_iter.next()
        except:
            labeled_iter = iter(labeled_trainloader)
            (_, inputs_x_s, inputs_x), targets_x = labeled_train_iter.next()
        try:
            (inputs_u_w, inputs_u_s, _), _ = unlabeled_train_iter.next()
        except: 
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s, _), _ = unlabeled_train_iter.next()
        try:
            (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()
        except:
            unlabeled_all_iter = iter(unlabeled_trainloader_all)
            (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()
        data_time.update(time.time() - end)

        b_size = inputs_x.shape[0]
        
        inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
        inputs = torch.cat([inputs_x, inputs_x_s,
                            inputs_all], 0).cuda()
        targets_x = targets_x.cuda()
        ## Feed data
        logits, logits_open = model(inputs)
        logits_open_u1, logits_open_u2 = logits_open[2*b_size:].chunk(2)

        ## Loss for labeled samples
        Lx = F.cross_entropy(logits[:2*b_size],
                                    targets_x.repeat(2), reduction='mean')
        Lo = ova_loss(logits_open[:2*b_size], targets_x.repeat(2))

        ## Open-set entropy minimization
        L_oem = ova_ent(logits_open_u1) / 2.
        L_oem += ova_ent(logits_open_u2) / 2.

        ## Soft consistenty regularization
        logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
        logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
        logits_open_u1 = F.softmax(logits_open_u1, 1)
        logits_open_u2 = F.softmax(logits_open_u2, 1)
        L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
            logits_open_u1 - logits_open_u2)**2, 1), 1))

        if epoch >= start_fix:
            inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).cuda()
            logits, logits_open_fix = model(inputs_ws)
            logits_u_w, logits_u_s = logits.chunk(2)
            pseudo_label = torch.softmax(logits_u_w.detach()/ T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(threshold).float()
            L_fix = (F.cross_entropy(logits_u_s,
                                        targets_u,
                                        reduction='none') * mask).mean()
            mask_probs.update(mask.mean().item())

        else:
            L_fix = torch.zeros(1).cuda().mean()
        loss = Lx + Lo + lambda_oem * L_oem  \
                + lambda_socr * L_socr + L_fix
        if  amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_o.update(Lo.item())
        losses_oem.update(L_oem.item())
        losses_socr.update(L_socr.item())
        losses_fix.update(L_fix.item())


        # record loss
        # losses.update(loss.item(), inputs_x.size(0))
        # losses_x.update(Lx.item(), inputs_x.size(0))
        # losses_u.update(Lu.item(), inputs_x.size(0)) 

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
 
        if (i+1)%cfg.SHOW_STEP==0:
            logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(epoch,i+1,total_step,losses.val,losses_x.avg,losses_u.val))
     
    return (losses.avg, losses_x.avg, losses_u.avg,)
 
 
def main_openmatch(cfg):
  # ===========  build logger =========== 
    logger, log_file = create_logger(cfg)
    
    # ===========  prepare output path ========
    path,model_dir,pic_dir =prepare_output_path(cfg,logger)
    
    # ===========  build model ============
    num_classes=cfg.DATASET.NUM_CLASSES
    use_norm = True if cfg.MODEL.LOSS.LABELED_LOSS == 'LDAM' else False
    model = models.__dict__[cfg.MODEL.NAME](cfg)
    model=model.cuda()
    cudnn.benchmark = True
    logger.info('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
   
    
    # ==========   build optimizer =========== 
    
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
     
    start_epoch ,best_epoch=1,1
    best_acc = 0
    # ==========   build dataloader ===========
    print('==> Preparing dataset {}'.format(cfg.DATASET.NAME))
    dataloaders=build_dataloader(cfg,logger)
    domain_trainloader=dataloaders[0]
    labeled_trainloader=dataloaders[1]
    unlabeled_trainloader=dataloaders[2]
    val_loader=dataloaders[3]
    test_loader=dataloaders[4]
    dual_labeled_trainloader=dataloaders[5] 
     # ==========   build criterion ==========
    dl_dist=labeled_trainloader.dataset.num_per_cls_list
    class_weight=get_class_weight(dl_dist,class_weight_type=cfg.MODEL.LOSS.LABELED_LOSS_CLASS_WEIGHT_TYPE)
    l_criterion,ul_criterion,val_criterion = build_loss(cfg)
    l_criterion.class_weight=class_weight.cuda()
    # ========== resume model ===============
    resume=cfg.RESUME
    if resume!='':
        # Load checkpoint.
        resume_file=cfg.RESUME
        assert os.path.isfile(resume_file), 'Error: no checkpoint directory found!'
        logger.info('==> Resuming from checkpoint..')
        output = os.path.dirname(resume_file)
        checkpoint = torch.load(resume_file)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer']) 
        logger.info('==> Resuming from epoch : {} done!'.format(start_epoch))        
    else:
        logger.info('Training the model from scratch!')
    
         
     
    max_epoch=cfg.MAX_EPOCH
    test_accs=[]
    test_group_accs=[]
    val_group_accs=[]
    best_val_acc_test_acc=0
    train_losses=[]
    for epoch in range(start_epoch, max_epoch+1):
        
        start_time = time.time() 
        logger.info('== Train Epoch: [{}|{}]'.format(epoch, max_epoch))

        train_loss, train_loss_x, train_loss_u= \
            train(labeled_trainloader, unlabeled_trainloader, 
                model, optimizer,  l_criterion,ul_criterion,
                    epoch,logger=logger,cfg=cfg,
                    dual_labeled_trainloader=dual_labeled_trainloader)
        scheduler.step()  
        val_loss, val_acc,val_group_acc = validate(val_loader,model, val_criterion, epoch=epoch, mode='Valid Stats',cfg=cfg)
        test_loss, test_acc ,test_group_acc= validate(test_loader,model, val_criterion,epoch=epoch, mode='Test Stats ',cfg=cfg)

        
        end_time = time.time()
        test_group_accs.append(test_group_acc)
        val_group_accs.append(val_group_acc)
        train_losses.append(train_loss)
        # append logger file 
        logger.info('== Train_loss:{:>5.4f}  train_loss_x:{:>5.4f}   train_loss_u:{:>5.4f}  val_losss:{:>5.4f}   test_loss:{:>5.4f}   Epoch_Time:{:>5.2f}min'.\
                        format(train_loss, train_loss_x, train_loss_u, val_loss, test_loss,(end_time - start_time) / 60))
        logger.info('== Val  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(val_group_acc[0]*100,val_group_acc[1]*100,val_group_acc[2]*100))
        logger.info('== Test group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(test_group_acc[0]*100,test_group_acc[1]*100,test_group_acc[2]*100))
        logger.info('== Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(val_acc*100,test_acc*100))
        
        # save model
         
        if val_acc > best_acc:
                best_acc, best_epoch, best_val_acc_test_acc = val_acc, epoch, test_acc
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch, 
                    'acc': val_acc,
                    'best_acc': best_acc, 
                    'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "best_model.pth") )
        
        logger.info(
                "--------------Best_Epoch:{:>3d}    Best_Val_Acc:{:>5.2f}  Test_Acc:{:>5.2f} --------------".format(
                    best_epoch, best_acc * 100, best_val_acc_test_acc*100
                )
            ) 
         
        test_accs.append(test_acc)
        logger.info('='*100)
         
    logger.info('Mean test acc:{:>5.2f}'.format(np.mean(test_accs[-20:]))) 
    # 画出训练过程中每个分组的acc变化折线图 
    plot_group_acc_over_epoch(group_acc=test_group_accs,save_path=os.path.join(pic_dir,'test_group_acc.jpg'))
    plot_group_acc_over_epoch(group_acc=val_group_accs,save_path=os.path.join(pic_dir,'val_group_acc.jpg'))
    plot_loss_over_epoch(train_losses,save_path=os.path.join(pic_dir,'train_loss.jpg'))
     


import random
seed=7
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True   
if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    # set random seed
    IF=[100]
    ood_r=[0.5,0.25,0.75]
    for if_ in IF:   
        for r in ood_r:  
            cfg.defrost()
            cfg.SEED=seed
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.freeze()
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            # main_openmatch(cfg)    
            trainer=OpenMatchTrainer(cfg)
            trainer.train()         
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
     
   
    
    
    
    

