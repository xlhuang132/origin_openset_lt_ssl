from __future__ import print_function
import logging
import argparse
import os
import shutil
import time
import random
from utils.set_seed import set_seed
import numpy as np
from time import sleep
from dataset.build_dataloader import build_dataloader
from utils.build_optimizer import get_optimizer, get_scheduler
from config.defaults import update_config,_C as cfg
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import copy
from utils.validate_model import validate
from utils import AverageMeter, accuracy, create_logger,\
                    plot_group_acc_over_epoch,prepare_output_path,\
                    interleave,plot_loss_over_epoch,FusionMatrix,plot_acc_over_epoch
import models  
import random
from skimage.filters import threshold_otsu
from utils.ema_model import EMAModel
from trainer import MTCFTrainer

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
    
 

def main_mtcf(cfg): 
    
    # ============ build logger ==============
    logger, log_file = create_logger(cfg)
    # ===========  prepare output path ========
    path,model_dir,pic_dir =prepare_output_path(cfg,logger)
    # ==========   build dataloader ===========
    print('==> Preparing dataset {}'.format(cfg.DATASET.NAME))
    dataloaders=build_dataloader(cfg,logger)
    domain_trainloader=dataloaders[0]
    labeled_trainloader=dataloaders[1]
    unlabeled_trainloader=dataloaders[2]
    val_loader=dataloaders[3]
    test_loader=dataloaders[4]
    dual_labeled_trainloader=dataloaders[5] 
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
    
    l_criterion = SemiLoss()
    val_criterion = nn.CrossEntropyLoss() 
    optimizer = get_optimizer(cfg, model)
     
    start_epoch = 1
    best_val_acc_test_acc=0 
    best_acc=0 
    # Resume 
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
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict']) 
        ema_model.load_state_dict(checkpoint['ema_state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer']) 
        logger.info('==> Resuming from epoch : {} done!'.format(start_epoch))        
    else:
        logger.info('Training the model from scratch!')
    
    # scheduler = get_scheduler(cfg, optimizer)
     
    
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
    
    
    if cfg.ALGORITHM.PRE_TRAIN.ENABLE:
        warmup_epoch=cfg.ALGORITHM.PRE_TRAIN.WARMUP_EPOCH      
        if start_epoch>warmup_epoch: pass
        else:
            for epoch in range(start_epoch,warmup_epoch+1):
                start_time = time.time() 
                logger.info('==Warmup Train Epoch: [{}|{}]'.format(epoch, warmup_epoch))
                train_acc,train_group_acc,train_loss = domain_train(domain_trainloader, model, optimizer, epoch ,logger,ema_model=ema_model)
 
                val_loss, val_acc,val_group_acc = validate(val_loader,model, val_criterion,epoch=epoch, mode='Valid Stats',cfg=cfg)
                test_loss, test_acc ,test_group_acc= validate(test_loader,model, val_criterion, epoch=epoch, mode='Test Stats ',cfg=cfg)
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
                logger.info('== Train_loss:{:>5.4f} val_losss:{:>5.4f}   test_loss:{:>5.4f}   Epoch_Time:{:>5.2f}min'.\
                                format(train_loss, val_loss, test_loss,(end_time - start_time) / 60))
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
                
                test_accs.append(test_acc)
                
            logger.info('='*100)
            # Train and val
            start_epoch=warmup_epoch+1
    for epoch in range(start_epoch,max_epoch+1):
        start_time = time.time() 
        logger.info('== Train Epoch: [{}|{}]'.format(epoch, max_epoch))

        train_acc,train_group_acc,train_loss, train_loss_x, train_loss_u= train(domain_trainloader, labeled_trainloader, unlabeled_trainloader, model, optimizer, l_criterion, epoch,logger,ema_model=ema_model,dual_labeled_trainloader=dual_labeled_trainloader)
        # scheduler.step()
        val_loss, val_acc,val_group_acc = validate(val_loader,ema_model, val_criterion, epoch=epoch,mode='Valid Stats',cfg=cfg)
        test_loss, test_acc ,test_group_acc= validate(test_loader,ema_model, val_criterion, epoch=epoch, mode='Test Stats ',cfg=cfg)
        
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
          
    plot_group_acc_over_epoch(group_acc=train_group_accs,title="Train Group Average Accuracy",save_path=os.path.join(pic_dir,'train_group_acc.jpg'),warmup_epoch=warmup_epoch)
    plot_group_acc_over_epoch(group_acc=val_group_accs,title="Val Group Average Accuracy",save_path=os.path.join(pic_dir,'val_group_acc.jpg'),warmup_epoch=warmup_epoch)
    plot_group_acc_over_epoch(group_acc=test_group_accs,title="Test Group Average Accuracy",save_path=os.path.join(pic_dir,'test_group_acc.jpg'),warmup_epoch=warmup_epoch)
    plot_acc_over_epoch(train_accs,title="Train average accuracy",save_path=os.path.join(pic_dir,'train_acc.jpg'),)
    plot_acc_over_epoch(test_accs,title="Test average accuracy",save_path=os.path.join(pic_dir,'test_acc.jpg'),)
    plot_acc_over_epoch(val_accs,title="Val average accuracy",save_path=os.path.join(pic_dir,'val_acc.jpg'),)
    plot_loss_over_epoch(train_losses,title="Train Average Loss",save_path=os.path.join(pic_dir,'train_loss.jpg'))
    plot_loss_over_epoch(val_losses,title="Val Average Loss",save_path=os.path.join(pic_dir,'val_loss.jpg'))
    plot_loss_over_epoch(test_losses,title="Test Average Loss",save_path=os.path.join(pic_dir,'test_loss.jpg'))
     

def domain_train(domain_trainloader, model, optimizer, epoch,logger,ema_model=None ):
    losses = AverageMeter()   
    model.train() 
    results = np.zeros((len(domain_trainloader.dataset)), dtype=np.float32)
    total_step=cfg.TRAIN_STEP
    data_iter=iter(domain_trainloader)
    for i in range(total_step): 
        try:
            data=data_iter.next()
        except:
            data_iter=iter(domain_trainloader)
            data=data_iter.next()
        inputs=data[0]
        targets=data[1]
        domain_labels=data[2] 
        indexs=data[3]
        inputs, domain_labels = inputs.cuda(), domain_labels.cuda(non_blocking=True)

        _,logits = model(inputs) 
        probs = torch.sigmoid(logits).view(-1)
        Ld = F.binary_cross_entropy_with_logits(logits, domain_labels.view(-1,1))

        results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

        loss = Ld

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        
        # current_lr = optimizer.param_groups[0]["lr"]
        # ema_decay =ema_model.update(model, step=i, current_lr=current_lr)
        if (i+1)%cfg.SHOW_STEP==0:
            logger.info('== Warmup Epoch:{} Step:[{}|{}] Avg_BCE_loss:{:>5.4f} =='.format(epoch,i+1,total_step,losses.val))
     

    domain_trainloader.dataset.label_update(results) 
    
    return (0.,[0.,0.,0.],losses.avg)

def train(domain_trainloader, labeled_trainloader, unlabeled_trainloader, 
          model, optimizer, l_criterion, epoch,logger, ema_model=None,
          dual_labeled_trainloader=None):
 
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()  
    
    fusion_matrix = FusionMatrix(cfg.DATASET.NUM_CLASSES)
    acc = AverageMeter()     
    func = torch.nn.Softmax(dim=1)
    
    labeled_train_iter = iter(labeled_trainloader)
    
    train_iter = iter(domain_trainloader)
    # 所有的数据concat在一起
    results = np.zeros((len(domain_trainloader.dataset)), dtype=np.float32)

    # Get OOD scores of unlabeled samples
    n_labeled=len(labeled_trainloader.dataset)
    weights = domain_trainloader.dataset.soft_labels[n_labeled:].copy()

    # Calculate threshold by otsu
    th = threshold_otsu(weights.reshape(-1,1))

    # Select samples having small OOD scores as ID data
    '''
    Attention:
    Weights is the (1 - OOD score) in this implement, which is different from the paper.
    So a larger weight means the data is more likely to be ID data.
    '''
    subset_indexs = np.arange(len(unlabeled_trainloader.dataset))[weights>=th]

    sub_target = unlabeled_trainloader.dataset.targets[subset_indexs]

    prec = (sub_target>=0).sum()/len(sub_target)

    recall = (sub_target>=0).sum()/((unlabeled_trainloader.dataset.targets>=0).sum())

    nsamples = len(subset_indexs) / len(unlabeled_trainloader.dataset)

    bs=cfg.DATASET.BATCH_SIZE
    num_workers=cfg.DATASET.NUM_WORKERS
    unlabeled_trainloader = data.DataLoader(data.Subset(unlabeled_trainloader.dataset, subset_indexs), batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)

    unlabeled_train_iter = iter(unlabeled_trainloader)

    dual_labeled_train_iter=None
    if dual_labeled_trainloader!=None:
        dual_labeled_train_iter=iter(dual_labeled_trainloader)
    total_step=cfg.TRAIN_STEP
    model.train()
    for i in range(total_step):
        try:
            inputs, targets, domain_labels, indexs = train_iter.next()
        except:
            train_iter = iter(domain_trainloader)
            inputs, targets, domain_labels, indexs = train_iter.next()
            
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        #  
        dual_inputs_x=None
        if dual_labeled_train_iter!=None:
            try: 
                dual_inputs_x, dual_targets_x=dual_labeled_train_iter.next()
            except:        
                dual_labeled_train_iter=iter(dual_labeled_trainloader) 
                dual_inputs_x, dual_targets_x=dual_labeled_train_iter.next()
            inputs_x=torch.cat((inputs_x,dual_inputs_x),dim=0)
            targets_x=torch.cat((targets_x,dual_targets_x),dim=0)
            inputs_u=torch.cat((inputs_u,inputs_u),dim=0)
            inputs_u2=torch.cat((inputs_u2,inputs_u2),dim=0)

        batch_size = inputs_x.size(0)
        if batch_size<inputs_u.size(0):
            inputs_u,inputs_u2=inputs_u[:batch_size],inputs_u2[:batch_size]

        # Transform label to one-hot
        num_classes=cfg.DATASET.NUM_CLASSES
        original_targets_x=targets_x.long()
        targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x.view(-1,1), 1)
 
        inputs = inputs.cuda()
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        domain_labels = domain_labels.cuda()

        model.apply(set_bn_eval)
        _,logits = model(inputs)
        model.apply(set_bn_train)

        probs = torch.sigmoid(logits).view(-1)
        Ld = F.binary_cross_entropy_with_logits(logits, domain_labels.view(-1,1))

        results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u,_ = model(inputs_u)
            outputs_u2,_ = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/cfg.ALGORITHM.MTCF.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(cfg.ALGORITHM.MTCF.MIXUP_ALPHA,cfg.ALGORITHM.MTCF.MIXUP_ALPHA)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        
        logits = [model(mixed_input[0])[0]]
        for input in mixed_input[1:]:
            logits.append(model(input)[0])

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        # compute 1st branch accuracy
        score_result = func(logits_x)
        now_result = torch.argmax(score_result, 1) 
        fusion_matrix.update(now_result.cpu().numpy(),original_targets_x.numpy())
        now_acc, cnt = accuracy(now_result.cpu().numpy(),original_targets_x.numpy())
        
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = l_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+i/total_step)

        loss = Ld + Lx + w * Lu

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
     
        
    
    domain_trainloader.dataset.label_update(results)
    group_acc=fusion_matrix.get_group_acc(cfg.DATASET.GROUP_SPLITS)
     
    return acc.avg, group_acc,losses.avg, losses_x.avg, losses_u.avg
 
def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch,lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean(torch.mean((probs_u - targets_u)**2, dim=1))

        return Lx, Lu, lambda_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999,lr=0.002):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model =copy.deepcopy(model).cuda()
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = False

def set_bn_train(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.track_running_stats = True

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
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
if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    # set random seed
    
    IF=[100] 
    ood_r=[0.75]  
 
    for if_ in IF:  # if
        # 同分布
        for r in ood_r: # r 
            cfg.defrost()
            cfg.DATASET.DL.IMB_FACTOR_L=if_
            cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.SEED=seed
            cfg.freeze()            
            print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
            trainer=MTCFTrainer(cfg)
            trainer.train()        
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        