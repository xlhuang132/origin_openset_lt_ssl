
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
    

def train(labeled_trainloader, unlabeled_trainloader, \
            model, optimizer, l_criterion,ul_criterion, \
            epoch,logger,cfg,ema_model=None,
            dual_labeled_trainloader=None):
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter() 
    fusion_matrix = FusionMatrix(cfg.DATASET.NUM_CLASSES)
    
    acc = AverageMeter()  
    
    
    func = torch.nn.Softmax(dim=1)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()
    total_step=cfg.TRAIN_STEP
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
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
        
        # DL 分类损失
        logits_x=model(inputs_x)
        Lx=l_criterion(logits_x, targets_x)
         # compute 1st branch accuracy
        score_result = func(logits_x)
        now_result = torch.argmax(score_result, 1) 
        fusion_matrix.update(now_result.cpu().numpy(), targets_x.cpu().numpy())
        now_acc, cnt = accuracy(now_result.cpu().numpy(),targets_x.cpu().numpy())
        # 弱增强伪标签
        with torch.no_grad(): 
            outputs_u = model(inputs_u)  
            p = outputs_u.softmax(dim=1)   
            confidence, pred_class = torch.max(p, dim=1)
        # 过滤低置信度标签
        loss_w=confidence.ge(cfg.ALGORITHM.CONFIDENCE_THRESHOLD ).float()
        logits_u = model(inputs_u2) 
        Lu = ul_criterion(logits_u, pred_class,weight=loss_w)  # weight对样本加权 class_weight对类加权 loss_weight对整个loss加权
        loss = Lx +  Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0)) 

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        current_lr = optimizer.param_groups[0]["lr"]
        ema_decay =ema_model.update(model, step=i, current_lr=current_lr)
 
        if (i+1)%cfg.SHOW_STEP==0:
            logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(epoch,i+1,total_step,losses.val,losses_x.avg,losses_u.val))
    group_acc=fusion_matrix.get_group_acc(cfg.DATASET.GROUP_SPLITS) 
    return (acc.avg, group_acc,losses.avg, losses_x.avg, losses_u.avg,)
 
 
def main_fixmatch(cfg):
  # ===========  build logger =========== 
    logger, log_file = create_logger(cfg)
    
    # ===========  prepare output path ========
    path,model_dir,pic_dir =prepare_output_path(cfg,logger)
    
    # ===========  build model ============
    num_classes=cfg.DATASET.NUM_CLASSES
    use_norm = True if cfg.MODEL.LOSS.LABELED_LOSS == 'LDAM' else False
    model = models.__dict__[cfg.MODEL.NAME](cfg)
    ema_model = EMAModel(
            model,
            cfg.MODEL.EMA_DECAY,
            cfg.MODEL.EMA_WEIGHT_DECAY, 
        )
    model=model.cuda()
    ema_model=ema_model.cuda()
    cudnn.benchmark = True
    logger.info('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
   
    
    # ==========   build optimizer =========== 
    
    optimizer = get_optimizer(cfg, model)
    # scheduler = get_scheduler(cfg, optimizer)
     
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
        best_epoch = checkpoint['best_epoch']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict']) 
        ema_model.load_state_dict(checkpoint['ema_state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer']) 
        logger.info('==> Resuming from epoch : {} done!'.format(start_epoch))        
    else:
        logger.info('Training the model from scratch!')
    
         
     
    max_epoch=cfg.MAX_EPOCH
    # accuracy
    test_accs=[]
    val_accs=[]
    train_accs=[]
    test_group_accs=[]
    val_group_accs=[]
    train_group_accs=[]
    
    # loss
    train_losses=[]
    val_losses=[]
    test_losses=[]
    best_val_acc_test_acc=0 
    for epoch in range(start_epoch, max_epoch+1):
        
        start_time = time.time() 
        logger.info('== Train Epoch: [{}|{}]'.format(epoch, max_epoch))

        train_acc,train_group_acc,train_loss, train_loss_x, train_loss_u= \
            train(labeled_trainloader, unlabeled_trainloader, 
                model, optimizer,  l_criterion,ul_criterion,
                    epoch,logger=logger,cfg=cfg,ema_model=ema_model,
                    dual_labeled_trainloader=dual_labeled_trainloader)
        # scheduler.step()  
        val_loss, val_acc,val_group_acc = validate(val_loader,ema_model, val_criterion, epoch=epoch, mode='Valid Stats',cfg=cfg)
        test_loss, test_acc ,test_group_acc= validate(test_loader,ema_model, val_criterion,epoch=epoch, mode='Test Stats ',cfg=cfg)

        
        end_time = time.time()
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_group_accs.append(train_group_acc)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_group_accs.append(val_group_acc)
        
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_group_accs.append(test_group_acc) 
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
                    'ema_state_dict': ema_model.state_dict(),
                    'epoch': epoch, 
                    'acc': val_acc,
                    'best_acc': best_acc, 
                    'best_epoch': best_epoch, 
                    'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "best_model.pth") )
        
        logger.info(
                "--------------Best_Epoch:{:>3d}    Best_Val_Acc:{:>5.2f}  Test_Acc:{:>5.2f} --------------".format(
                    best_epoch, best_acc * 100, best_val_acc_test_acc*100
                )
            ) 
        if epoch % cfg.SAVE_EPOCH == 0: 
            torch.save({
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'epoch': epoch, 
                    'acc': val_acc,
                    'best_acc': best_acc,                     
                    'best_epoch': best_epoch, 
                    'optimizer': optimizer.state_dict()
                }, os.path.join(
                model_dir,
                "epoch_{}.pth".format(epoch),
            ))
        test_accs.append(test_acc)
        logger.info('='*100)
         
    logger.info('Mean test acc:{:>5.2f}'.format(np.mean(test_accs[-20:]))) 
    # 画出训练过程中每个分组的acc变化折线图 
    plot_group_acc_over_epoch(group_acc=train_group_accs,title="Train Group Average Accuracy",save_path=os.path.join(pic_dir,'train_group_acc.jpg'))
    plot_group_acc_over_epoch(group_acc=val_group_accs,title="Val Group Average Accuracy",save_path=os.path.join(pic_dir,'val_group_acc.jpg'))
    plot_group_acc_over_epoch(group_acc=test_group_accs,title="Test Group Average Accuracy",save_path=os.path.join(pic_dir,'test_group_acc.jpg'))
    plot_acc_over_epoch(train_accs,title="Train average accuracy",save_path=os.path.join(pic_dir,'train_acc.jpg'),)
    plot_acc_over_epoch(test_accs,title="Test average accuracy",save_path=os.path.join(pic_dir,'test_acc.jpg'),)
    plot_acc_over_epoch(val_accs,title="Val average accuracy",save_path=os.path.join(pic_dir,'val_acc.jpg'),)
    plot_loss_over_epoch(train_losses,title="Train Average Loss",save_path=os.path.join(pic_dir,'train_loss.jpg'))
    plot_loss_over_epoch(val_losses,title="Val Average Loss",save_path=os.path.join(pic_dir,'val_loss.jpg'))
    plot_loss_over_epoch(test_losses,title="Test Average Loss",save_path=os.path.join(pic_dir,'test_loss.jpg'))
    

    
if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    # set random seed
    set_seed(cfg.SEED)
    cudnn.benchmark = True 
    IF=[100]
    ood_r=[0.0,0.5,0.75]
    for if_ in IF:   
        for r in ood_r:  
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.freeze()
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            main_fixmatch(cfg)            
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
     
   
    
    
    
    

