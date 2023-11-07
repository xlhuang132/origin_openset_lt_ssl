
 
import logging 
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
import os   
import torch.nn.functional as F
from utils import AverageMeter, accuracy, create_logger,plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch
from utils.build_optimizer import get_optimizer, get_scheduler
from utils.utils import linear_rampup
from utils.validate_model import validate
from utils.mixup import mix_up
from utils.set_seed import set_seed
from trainer import SupervisedTrainer
def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/baseline_cifar10.yaml",
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
    

def train(labeled_trainloader, unlabeled_trainloader, 
          model, optimizer, l_criterion,ul_criterion, 
          epoch,logger,cfg,
          dual_labeled_trainloader=None):
    losses = AverageMeter()
    losses_x = AverageMeter() 
    losses_u = AverageMeter()
    model.train() 
    total_step=cfg.TRAIN_STEP
    labeled_train_iter = iter(labeled_trainloader)
    # unlabeled_train_iter = iter(unlabeled_trainloader)
    dual_labeled_train_iter=None
    if dual_labeled_trainloader!=None:
        dual_labeled_train_iter=iter(dual_labeled_trainloader)
        
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
        
        # # DU 数据
        # try:
        #     (inputs_u, inputs_u2), _, = unlabeled_train_iter.next()
        # except:
        #     unlabeled_train_iter = iter(unlabeled_trainloader)
        #     (inputs_u, inputs_u2), _, = unlabeled_train_iter.next()
        # # 限制两批数量相等
        # bs=inputs_x.size(0)
        # if bs<inputs_u.size(0):
        #     inputs_u,inputs_u2=inputs_u[:bs],inputs_u2[:bs]

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        # inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()  
        # 是否进行mixup
        # if cfg.ALGORITHM.BRANCH1_MIXUP:
        #     inputs_x=mix_up(inputs_x, inputs_u)
        if dual_inputs_x!=None:
            dual_inputs_x, dual_targets_x = dual_inputs_x.cuda(), dual_targets_x.cuda(non_blocking=True)
            # if cfg.ALGORITHM.BRANCH2_MIXUP:
                # dual_inputs_x=mix_up(dual_inputs_x, inputs_u)
            
            dual_logits_x=model(dual_inputs_x)
            dual_lx=l_criterion(dual_logits_x, dual_targets_x.long())
            loss+=dual_lx
            losses.update(dual_lx.item(), dual_inputs_x.size(0)) 
                
        # DL 分类损失
        logits_x=model(inputs_x)
        lx=l_criterion(logits_x, targets_x.long())
        losses_x.update(lx.item(),inputs_x.size(0))
        loss+=lx
        losses.update(lx.item(), inputs_x.size(0)) 
        # record loss
         
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
 
        if (i+1)%cfg.SHOW_STEP==0:
            logger.info('== Epoch:{} Step:[{}|{}] Avg_Loss_x:{:>5.4f}  =='.format(epoch,i+1,total_step,losses_x.avg))
     
    return (losses.avg, losses_x.avg, losses_u.avg,)
 
 
def main_baseline(cfg):
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
    # ==========   build criterion ==========
    l_criterion,ul_criterion,val_criterion = build_loss(cfg)
    
    # ==========   build optimizer =========== 
    
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
     
    start_epoch = 1 
    best_epoch=1
    best_acc = 0
    # ==========   build dataloader ===========
    print('==> Preparing dataset {}'.format(cfg.DATASET.NAME))
    _,labeled_trainloader,unlabeled_trainloader,val_loader,test_loader,dual_labeled_trainloader=build_dataloader(cfg,logger)
    
    
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
    train_losses=[]
    best_val_acc_test_acc=0
    for epoch in range(start_epoch, max_epoch+1):
        
        start_time = time.time() 
        logger.info('== Train Epoch: [{}|{}]'.format(epoch, max_epoch))

        train_loss, train_loss_x, train_loss_u= \
            train(labeled_trainloader, unlabeled_trainloader,
                  model, optimizer,  l_criterion,ul_criterion,
                  epoch,logger=logger,cfg=cfg,
                  dual_labeled_trainloader=dual_labeled_trainloader)
        scheduler.step()  
        train_losses.append(train_loss)
        val_loss, val_acc,val_group_acc = validate(val_loader,model, val_criterion, epoch=epoch, mode='Valid Stats',cfg=cfg)
        test_loss, test_acc ,test_group_acc= validate(test_loader,model, val_criterion, epoch=epoch, mode='Test Stats ',cfg=cfg)

        
        end_time = time.time()
        test_group_accs.append(test_group_acc)
        val_group_accs.append(val_group_acc)
        
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
        if epoch % cfg.SAVE_EPOCH == 0:
            model_save_path = os.path.join(
                model_dir,
                "epoch_{}.pth".format(epoch),
            )
            torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch, 
                    'acc': val_acc,
                    'best_acc': best_acc, 
                    'optimizer': optimizer.state_dict()
                }, model_save_path)
         
        test_accs.append(test_acc)
        logger.info('='*100)
         
    logger.info('Mean test acc:{:>5.2f}'.format(np.mean(test_accs[-20:]))) 
    
    # 画出训练过程中每个分组的acc变化折线图 
    plot_group_acc_over_epoch(group_acc=test_group_accs,save_path=os.path.join(pic_dir,'test_group_acc.jpg'))
    plot_group_acc_over_epoch(group_acc=val_group_accs,save_path=os.path.join(pic_dir,'val_group_acc.jpg'))
    plot_loss_over_epoch(train_losses,save_path=os.path.join(pic_dir,'train_loss.jpg'))
      
if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    # set random seed
    set_seed(cfg.SEED)
    cudnn.benchmark = True 
    IF=[50] # 10,50,100
    ood_r=[0.0,] # basline不用mixup的话不用考虑r 0.0,0.25, 0.5, 0.75,1.0 randomsampler+classreversedsampler没有用到mixup
    for if_ in IF:  # if
        # 同分布
        for r in ood_r:  
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            cfg.freeze() 
            trainer=SupervisedTrainer(cfg)
            trainer.train()
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        