
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

from trainer import MOODTrainer

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

def update_center_adaptive_thres(features,logits,labels,center,num_count,conf_thres):
    with torch.no_grad():
        p = logits.detach().softmax(dim=1)  # soft pseudo labels
        confidence, pred_class = torch.max(p, dim=1)
        # 置信度大于阈值
        conf_thr=conf_thres[labels]
        ones=torch.ones_like(conf_thr).cuda()
        zeros=torch.zeros_like(conf_thr).cuda()
        selected_index = torch.where(confidence > conf_thr,ones,zeros).cuda() 
        # 同时分类正确
        ones=torch.ones_like(labels).cuda()
        zeros=torch.zeros_like(labels).cuda()
        corr = torch.where(labels == pred_class,ones,zeros).cuda() 
        selected_index *=corr
        index= torch.nonzero(selected_index , as_tuple=False).squeeze(1)
        if index.shape[0]==0:
            return center,num_count,conf_thres
        embedding=features[index].detach()
        final_y=labels[index].detach()
        if len(final_y) > 0:
            uniq_c = torch.unique(final_y)
            for c in uniq_c:
                c = int(c)
                select_index = torch.nonzero(
                    final_y == c, as_tuple=False).squeeze(1)
                embedding_temp = embedding[select_index] 
                confidence_mean=confidence[select_index].mean(dim=0)
                mean = embedding_temp.mean(dim=0)
                var = embedding_temp.var(dim=0, unbiased=False)
                n = embedding_temp.numel() / embedding_temp.size(1)
                if n > 1:
                    var = var * n / (n - 1)
                else:
                    var = var 
                if num_count[c] > 0: 
                    center[c] = 0.9 * mean + (
                            1 - 0.9) * center[c]
                    conf_thres[c]=0.9*confidence_mean+(1-0.9)*conf_thres[c] 
                else:
                    center[c] = mean 
                    conf_thres[c]= confidence_mean
                num_count[c] += len(embedding_temp)
    return center,num_count,conf_thres

def update_center(embedding,labels,center,num_count,cfg):
    decay_ratio=cfg.MODEL.LOSS.FEATURE_LOSS.CENTER_DECAY_RATIO
    with torch.no_grad():
        if len(labels) > 0:
            uniq_c = torch.unique(labels)
            for c in uniq_c:
                c = int(c)
                select_index = torch.nonzero(
                    labels == c, as_tuple=False).squeeze(1)
                embedding_temp = embedding[select_index]  
                mean = embedding_temp.mean(dim=0)
                var = embedding_temp.var(dim=0, unbiased=False)
                n = embedding_temp.numel() / embedding_temp.size(1)
                if n > 1:
                    var = var * n / (n - 1)
                else:
                    var = var 
                if num_count[c] > 0: 
                    center[c] =decay_ratio* mean + (
                            1 - decay_ratio) * center[c] 
                else:
                    center[c] = mean  
                num_count[c] += len(embedding_temp)
    return center,num_count 

def compute_feature_dist(feature,feature_center): 
    return  torch.cdist(feature,feature_center) # 与各个中心之间的欧式距离

def get_feature_dist_loss(features,targets,dl_center,id_mask,cfg):
    """
        feature_x + feature_u *id_mask:向对应的dl_center靠近
        feature_u*(1-id_mask):远离dl_center
    """ 
    temperature=cfg.MODEL.LOSS.FEATURE_LOSS.TEMPERATURE
    pair_dist=-1  *torch.cdist(features,dl_center) # old
    # pair_dist=-1  *torch.cdist(dl_center,features) # new
    # [B, K] 
    logits=torch.div(pair_dist, temperature) 
    num_classes=dl_center.size(0) 
    # mask_same_c :torch.Size([192, 10]) old
    mask_same_c=torch.eq(\
        targets.contiguous().view(-1, 1).cuda(), \
        torch.tensor([i for i in range(num_classes)]).contiguous().view(-1, 1).cuda().T).float()
    
    id_mask=id_mask.expand(mask_same_c.size(1),-1).T # torch.Size([10,192]) # old
    # mask_same_c :torch.Size([10, 192]) new
    # mask_same_c=torch.eq(\
    #     torch.tensor([i for i in range(num_classes)]).contiguous().view(-1, 1).cuda(),
    #     targets.contiguous().view(-1, 1).cuda().T).float()
    # id_mask=id_mask.expand(mask_same_c.size(0),-1) # torch.Size([192, 10]) # new
    mask_same_c*=id_mask
    log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
    log_prob_pos = log_prob * mask_same_c # mask掉其他的负类
    loss = - log_prob_pos.sum() / mask_same_c.sum()   
    return loss 

def get_id_feature_dist_loss(features,targets,dl_center,id_mask,cfg):
    """
        feature_x + feature_u *id_mask:向对应的dl_center靠近
        feature_u*(1-id_mask):远离dl_center
    """ 
    temperature=cfg.MODEL.LOSS.FEATURE_LOSS.TEMPERATURE
    pair_dist=-1  *torch.cdist(features,dl_center)  
    logits=torch.div(pair_dist, temperature) 
    num_classes=dl_center.size(0) 
    mask_same_c=torch.eq(\
        targets.contiguous().view(-1, 1).cuda(), \
        torch.tensor([i for i in range(num_classes)]).contiguous().view(-1, 1).cuda().T).float()
    id_mask=id_mask.expand(mask_same_c.size(1),-1).T # torch.Size([10,192]) # old 
    mask_same_c*=id_mask
    log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
    log_prob_pos = log_prob * mask_same_c # mask掉其他的负类
    loss = - log_prob_pos.sum() / mask_same_c.sum()   
    return loss      

# def get_ood_feature_dist_loss(features_u,features_u2,dl_center,ood_mask,cfg):
#     """
#         feature_x + feature_u *id_mask:向对应的dl_center靠近
#         feature_u*(1-id_mask):远离dl_center
#     """  
#     # temperature=cfg.MODEL.LOSS.FEATURE_LOSS.TEMPERATURE
#     # pair_dist=-1 *torch.cdist(features_u,dl_center)  
    
#     # logits=torch.div(pair_dist, temperature)  
#     # log_prob = - torch.log((torch.exp(logits)).sum(1, keepdim=True))
#     # log_prob_pos = log_prob * ood_mask  
#     # loss = - log_prob_pos.sum() / ood_mask.sum()   
    
#     num_classes=dl_center.size(0) 
#     temperature=cfg.MODEL.LOSS.FEATURE_LOSS.TEMPERATURE
#     mask_same_c=torch.cat([torch.eye(features_u.size(0)),torch.eye(features_u.size(0)),torch.zeros((features_u.size(0),dl_center.size(0)))],dim=1)
#     mask_same_c=torch.cat([mask_same_c,mask_same_c],dim=0).cuda()
#     du_features=torch.cat([features_u,features_u2],dim=0)
#     temp_features=torch.cat([du_features,dl_center],dim=0)
    
#     pair_dist=-1 *torch.cdist(du_features,temp_features)  # [B,B+C]
    
#     logits=torch.div(pair_dist, temperature)  
#     ood_mask=torch.cat([ood_mask,ood_mask],dim=0)
#     ood_mask=ood_mask.expand(mask_same_c.size(1),-1).T
#     mask_same_c*=ood_mask
#     log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
#     log_prob_pos = log_prob * mask_same_c # mask掉其他的负类
#     loss = - log_prob_pos.sum() / mask_same_c.sum()   
#     return loss          
def get_ood_feature_dist_loss(features_u,features_u2,dl_center,ood_mask,cfg):
    """
        feature_x + feature_u *id_mask:向对应的dl_center靠近
        feature_u*(1-id_mask):远离dl_center
    """       
    num_classes=dl_center.size(0) 
    temperature=cfg.MODEL.LOSS.FEATURE_LOSS.TEMPERATURE
    mask_same_c=torch.cat([torch.eye(features_u.size(0)),torch.zeros((features_u.size(0),dl_center.size(0)))],dim=1).cuda()
    temp_features=torch.cat([features_u2,dl_center],dim=0)    
    pair_dist=-1 *torch.cdist(features_u,temp_features)     
    logits=torch.div(pair_dist, temperature)   
    ood_mask=ood_mask.expand(mask_same_c.size(1),-1).T
    mask_same_c*=ood_mask
    log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
    log_prob_pos = log_prob * mask_same_c # mask掉其他的负类
    loss = - log_prob_pos.sum() / mask_same_c.sum()   
    return loss          
   
def pre_train(model,data_loader,pretrain_optimizer,temperature,logger,show_step,epoch,cfg):
    model.train()
    total_loss, total_num = 0.0, 0 
    total_step=cfg.VAL_ITERATION
    data_iter=iter(data_loader)
    for i in range(total_step):
        try:
            data=data_iter.next()
        except:
            data_iter=iter(data_loader)
            data=data_iter.next()
        pos_1=data[0][0]
        pos_2=data[0][1]
        target=data[1] 
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        _,out_1 = model(pos_1,return_encoding=True)# feature:torch.Size([64, 128])
        _,out_2 = model(pos_2,return_encoding=True)# feature:torch.Size([64, 128])
        # out_1 = model(pos_1,return_encoding=True)# feature:torch.Size([64, 128])
        # out_2 = model(pos_2,return_encoding=True)# feature:torch.Size([64, 128])
 
        batch_size=out_1.size(0)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        if (i+1) %show_step==0:
            logger.info('== Epoch:{} Step:[{}|{}] Loss:{:>5.4f}  =='.format(epoch,i+1,total_step, total_loss / total_num))
     
    return total_loss / total_num ,0.,[0.,0.,0.]
 
def train(labeled_trainloader, unlabeled_trainloader, model, \
        optimizer, l_criterion,ul_criterion, epoch,logger,cfg,\
        dl_center,dl_num_count,dl_conf_thres,\
        # du_center,du_num_count,du_conf_thres, 
        dual_labeled_trainloader,feat,feat_y
        ):
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter() 
    losses_feat = AverageMeter() 
    
    losses_id_feat = AverageMeter() 
    losses_ood_feat = AverageMeter() 
    acc = AverageMeter()  
    
    fusion_matrix = FusionMatrix(cfg.DATASET.NUM_CLASSES)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()
    total_step=cfg.TRAIN_STEP
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    dual_labeled_train_iter=None
    w_feat=cfg.MODEL.LOSS.FEATURE_LOSS.WEIGHT
    w_id_feat=cfg.MODEL.LOSS.FEATURE_LOSS.ID_LOSS_WEIGHT
    w_ood_feat=cfg.MODEL.LOSS.FEATURE_LOSS.OOD_LOSS_WEIGHT
    if dual_labeled_trainloader!=None:
        dual_labeled_train_iter=iter(dual_labeled_trainloader)
    
    func = torch.nn.Softmax(dim=1)
    
    # ================== 第一个正式训练先计算dl_center，不使用feature loss =============
    if_use_feature_loss= not epoch==cfg.ALGORITHM.PRE_TRAIN.WARMUP_EPOCH+1
    for i in range(1,total_step+1):
        loss =0
        # DL 数据
        try:
            inputs_x, targets_x = labeled_train_iter.next() 
        except:
            labeled_train_iter = iter(labeled_trainloader)          
            inputs_x, targets_x = labeled_train_iter.next()
        inputs_dual_x=None
        if dual_labeled_train_iter!=None:
            try: 
                inputs_dual_x, targets_dual_x=dual_labeled_train_iter.next()
            except:        
                dual_labeled_train_iter=iter(dual_labeled_trainloader) 
                inputs_dual_x, targets_dual_x=dual_labeled_train_iter.next()
        
        # DU 数据
        try:
            (inputs_u, inputs_u2), _,idxs = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _,idxs = unlabeled_train_iter.next()
            
        # 限制两批数量相等
        bs=inputs_x.size(0)
        if bs<inputs_u.size(0):
            inputs_u,inputs_u2=inputs_u[:bs],inputs_u2[:bs]
        
        # 1、DU cons loss
        # 弱增强伪标签       
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        outputs_u,features_u = model(inputs_u,return_encoding=True)
        if not cfg.ALGORITHM.ABLATION.ENABLE \
            or cfg.ALGORITHM.ABLATION.ENABLE and cfg.ALGORITHM.ABLATION.OOD_DETECTION:            
            id_mask,ood_mask=knn_ood_detect(feat, feat_y, features_u, cfg) 
        with torch.no_grad(): 
            p = outputs_u.softmax(dim=1)   
            confidence, pred_class = torch.max(p, dim=1) 
        loss_w=confidence.ge(cfg.ALGORITHM.CONFIDENCE_THRESHOLD).float()
        # 是否进行OOD检测并与检测后的ood样本进行mixup
        if not cfg.ALGORITHM.ABLATION.ENABLE or cfg.ALGORITHM.ABLATION.ENABLE and cfg.ALGORITHM.ABLATION.OOD_DETECTION:
            loss_w*= id_mask # 与id_mask相乘
        logits_u,features_u2 = model(inputs_u2,return_encoding=True) 
        Lu = ul_criterion(logits_u, pred_class,weight=loss_w)  # weight对样本加权 class_weight对类加权 loss_weight对整个loss加权
        loss +=  Lu
        
        
        # 2、DL 分类损失         
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_dual_x, targets_dual_x = inputs_dual_x.cuda(), targets_dual_x.cuda(non_blocking=True)
        if cfg.ALGORITHM.BRANCH1_MIXUP:
            inputs_x=mix_up(inputs_x, inputs_u)
        if not cfg.ALGORITHM.ABLATION.ENABLE or cfg.ALGORITHM.ABLATION.ENABLE and cfg.ALGORITHM.ABLATION.MIXUP:
            if not cfg.ALGORITHM.ABLATION.ENABLE: # 正常训练
                ood_index = torch.nonzero(ood_mask, as_tuple=False).squeeze(1)  
                inputs_dual_x=mix_up(inputs_dual_x, inputs_u[ood_index],model=model,cfg=cfg)
            else: # 消融实验
                if cfg.ALGORITHM.ABLATION.MIXUP: 
                    if cfg.ALGORITHM.ABLATION.OOD_DETECTION:
                        ood_index = torch.nonzero(ood_mask, as_tuple=False).squeeze(1)  
                        inputs_dual_x=mix_up(inputs_dual_x, inputs_u[ood_index],model=model,cfg=cfg)            
                    else: 
                        inputs_dual_x=mix_up(inputs_dual_x, inputs_u,model=model,cfg=cfg)
        
        inputs_x=torch.cat([inputs_x,inputs_dual_x],dim=0)  
        targets_x=torch.cat([targets_x,targets_dual_x],dim=0)
        logits_x,features_x=model(inputs_x,return_encoding=True)
        Lx=l_criterion(logits_x, targets_x)
        loss+=Lx
        # compute 1st branch accuracy
        score_result = func(logits_x)
        now_result = torch.argmax(score_result, 1) 
        fusion_matrix.update(now_result.cpu().numpy(), targets_x.cpu().numpy())
        now_acc, cnt = accuracy(now_result.cpu().numpy(),targets_x.cpu().numpy())
        acc.update(now_acc, cnt)
        
        # 3、feature loss 使用有监督的对比学习
       
        if not cfg.ALGORITHM.ABLATION.ENABLE and if_use_feature_loss or cfg.ALGORITHM.ABLATION.ENABLE and cfg.ALGORITHM.ABLATION.FEAT_LOSS:
             
            # id feature loss
            all_features=torch.cat([features_x,features_u],dim=0)
            all_target=torch.cat([targets_x,pred_class],dim=0)
            n_x=targets_x.size(0)
            confidenced_id_mask= torch.cat([torch.ones(n_x).cuda(),id_mask*loss_w]).long() 
            Lidfeat=get_id_feature_dist_loss(all_features, all_target, dl_center, confidenced_id_mask, cfg)
            # ood feature loss            
            Loodfeat=0. 
            if not cfg.ALGORITHM.ABLATION.ENABLE or cfg.ALGORITHM.ABLATION.ENABLE and cfg.ALGORITHM.ABLATION.OOD_DETECTION:
              
                if ood_mask.sum()>0:      
                    Loodfeat=get_ood_feature_dist_loss(features_u,features_u2, dl_center, ood_mask, cfg) 
            Lfeat=w_id_feat* Lidfeat+ w_ood_feat* Loodfeat
            # Lfeat=get_feature_dist_loss(all_features,all_target,dl_center,id_mask,cfg) 

            if Lfeat.item()<0:
                logger.warning("Lfeat is {}".format(Lfeat.item()))
        else:
            Lfeat=0.
        loss+=w_feat*Lfeat
        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0)) 
        if Lfeat!=0:
            losses_feat.update(Lfeat.item(), all_features.size(0)) 
            losses_id_feat.update(Lidfeat.item(), id_mask.sum())
            if Loodfeat!=0:
                losses_ood_feat.update(Loodfeat.item(), all_features.size(0)-id_mask.sum())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        # dl_center,dl_num_count,dl_conf_thres=update_center_adaptive_thres(features_x, logits_x, targets_x, dl_center, dl_num_count,dl_conf_thres)
        dl_center,dl_num_count=update_center(features_x, targets_x, dl_center, dl_num_count,cfg)
        
        if (i+1)%cfg.SHOW_STEP==0:
            logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f}  =='.format(epoch,i+1,total_step,losses.val,losses_x.val,losses_u.val))
            logger.info('            Loss_feat:{:>5.4f} Loss_id_feat:{:>5.4f} Loss_ood_feat:{:>5.4f}              '.format(losses_feat.val,losses_id_feat.val,losses_ood_feat.val))
    
    group_acc=fusion_matrix.get_group_acc(cfg.DATASET.GROUP_SPLITS)
    return (acc.avg, group_acc,losses.avg, losses_x.avg, losses_u.avg,dl_center,dl_num_count,dl_conf_thres)
 
def simclr_create_memory_bank(model,labeled_trainloader,unlabeled_trainloader,cfg):
  
    model.eval()
    
    total_top1, total_num, feature_bank = 0.0,  0, []
    c=cfg.DATASET.NUM_CLASSES
    k=cfg.ALGORITHM.PRE_TRAIN.SimCLR.K
    temperature=cfg.ALGORITHM.PRE_TRAIN.SimCLR.TEMPERATURE 
    
    with torch.no_grad():
        # 有标签数据
        for i,(x,y) in enumerate(labeled_trainloader):
            x=x.cuda() 
            out,feature = model(x,return_encoding=True)
            feature_bank.append(feature) 
        feature_bank_labeled = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(labeled_trainloader.dataset.targets).cuda()
        
        for i,([data,_],_,_) in enumerate(unlabeled_trainloader):
            data=data.cuda() 
            out,feature = model(data,return_encoding=True) 
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank_labeled)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
             
    return 

def prepare_feat(model,domain_dataloader,feat_idx=None,feat=None,cfg=None):
    model.eval() 
    tmp_feat=[]
    domain_y=[]
    tmp_idx=[] 
    decay_ratio=cfg.ALGORITHM.OOD_DETECTOR.FEATURE_DECAY_RATIO
    with torch.no_grad():
            for i,data in enumerate(domain_dataloader):
                x=data[0] 
                if len(x)==2:
                    x=x[0]
                x=x.cuda()
                domain_labels=data[2]
                idx=data[3]
                _,feature=model(x,return_encoding=True)
                domain_y.append(domain_labels)
                tmp_feat.append(feature)
                tmp_idx.append(idx)
            tmp_feat = torch.cat(tmp_feat, dim=0).contiguous().cuda()
            domain_y =  torch.cat(domain_y, dim=0).t().long().cuda()
            tmp_idx=torch.cat(tmp_idx,dim=0).numpy().tolist()
    if feat_idx == None:
        feat_idx=tmp_idx
        feat=tmp_feat
    else: # 现在的idex和之前的index进行对应
        select_index=[]
        for idx in feat_idx:            
            select_idx=tmp_idx.index(idx)
            select_index.append(select_idx)
        select_index=torch.Tensor(select_index).long().cuda() 
        feat= (1-decay_ratio)*feat +decay_ratio* tmp_feat[select_index]
        domain_y=domain_y[select_index]                 
    return feat,domain_y,feat_idx
            
def knn_ood_detect(feat,feat_y,food,cfg):
    k=cfg.ALGORITHM.OOD_DETECTOR.K
    temperature=cfg.ALGORITHM.OOD_DETECTOR.TEMPERATURE
    sim_matrix = torch.mm(food, feat.t())
    sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1) # 
    
    # sim_labels = torch.gather(feat_y.expand(food.size(0), -1), dim=-1, index=sim_indices)
    # sim_weight = (sim_weight / temperature).exp()
    new_d_y=[]
    # for item in sim_indices: # [n,50] 
    d_y=feat_y[sim_indices]        
    count_idnn=torch.count_nonzero(d_y,dim=1)        
    ones=torch.ones_like(count_idnn).cuda()
    zeros=torch.zeros_like(count_idnn).cuda()
    id_mask = torch.where(count_idnn >= k//2,ones,zeros).long().cuda() 
    
    ood_mask = torch.where(count_idnn <= k//2,ones,zeros).long().cuda() 
    return id_mask,ood_mask

def pick_mixup_sample(dl,du,cfg):
    # k=cfg.ALGORITHM.OOD_DETECTOR.K
    temperature=cfg.ALGORITHM.OOD_DETECTOR.TEMPERATURE
    sim_matrix = torch.mm(dl, du.t())
    sim_weight, sim_indices = (-sim_matrix).topk(k=1, dim=-1) # 
    return du[sim_indices]

def update_domain_label(feat,feat_y,cfg): 
    k=cfg.ALGORITHM.OOD_DETECTOR.K    
    iteration=cfg.ALGORITHM.OOD_DETECTOR.UPDATE_ITER
    for i in range(iteration): # 迭代10次更新
        select_index=torch.nonzero(feat_y == 0, as_tuple=False).squeeze(1)
        if select_index.size(0)==0:
            break
        ood_feat=feat[select_index]
        # [B, K]        
        sim_matrix = torch.mm(ood_feat, feat.t())
        sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1) # 
        new_d_y=[]
        # for item in sim_indices: # [n,50] 
        d_y=feat_y[sim_indices]        
        count_idnn=torch.count_nonzero(d_y,dim=1)        
        ones=torch.ones_like(count_idnn).cuda()
        zeros=torch.zeros_like(count_idnn).cuda()
        new_d_y = torch.where(count_idnn >= k//2,ones,zeros).long().cuda() 
        feat_y[select_index]=new_d_y
    return feat_y

def main_ours(cfg):
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
     
    start_epoch ,best_epoch=1,1
    
    best_epoch=1
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
    pre_train_loader=dataloaders[6]
    # 有标签数据的分布
    dl_dist=labeled_trainloader.dataset.num_per_cls_list
   
    
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
        
    dl_center=torch.zeros(num_classes,cfg.ALGORITHM.PRE_TRAIN.SimCLR.FEATURE_DIM).cuda() 
    dl_num_count= torch.zeros(num_classes).cuda()
    dl_conf_thres=torch.zeros(num_classes).cuda() 
    
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
    warmup_epoch=cfg.ALGORITHM.PRE_TRAIN.WARMUP_EPOCH
    
    model,start_epoch,optimizer=load_warmup_model(model, cfg,optimizer)
    
    # scheduler = get_scheduler(cfg, optimizer) # WRN训练不用
    # 先使用CEloss进行预训练，class_weight用 cbloss
    # ==========   build criterion ========== 
    l_criterion,ul_criterion,val_criterion = build_loss(cfg)
         
    logger.info('== The class weight type of warm_up training is {}'.format(cfg.MODEL.LOSS.WARMUP_LABELED_LOSS_CLASS_WEIGHT_TYPE))
    logger.info('== Start warm_up trainging...')
    # 预训练优化器
    # pretrain_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    for epoch in range(start_epoch,warmup_epoch+1):
        start_time = time.time() 
        logger.info('==Warmup Train Epoch: [{}|{}]'.format(epoch, warmup_epoch)) 
        # train_loss=pre_train(model, pre_train_loader,pretrain_optimizer,cfg.ALGORITHM.PRE_TRAIN.SimCLR.TEMPERATURE,logger,cfg.SHOW_STEP,epoch,cfg)
        train_loss,train_acc,train_group_acc=pre_train(model, pre_train_loader,optimizer,cfg.ALGORITHM.PRE_TRAIN.SimCLR.TEMPERATURE,logger,cfg.SHOW_STEP,epoch,cfg)
        
        # scheduler.step()  
        val_loss, val_acc,val_group_acc = validate(val_loader,model, val_criterion,  epoch=epoch, mode='Valid Stats',cfg=cfg)
        test_loss, test_acc ,test_group_acc= validate(test_loader,model, val_criterion, epoch=epoch, mode='Test Stats ',cfg=cfg)
        end_time = time.time()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        val_accs.append(val_acc)
        train_group_accs.append(train_group_acc)
        test_group_accs.append(test_group_acc)
        val_group_accs.append(val_group_acc)
        # append logger file 
        logger.info('== Train_loss:{:>5.4f}  val_losss:{:>5.4f}   test_loss:{:>5.4f}   Epoch_Time:{:>5.2f}min'.\
                        format(train_loss,   val_loss, test_loss,(end_time - start_time) / 60))
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
          
        logger.info('='*100)
        if epoch==warmup_epoch:
            torch.save({
                            'state_dict': model.state_dict(),
                            'epoch': epoch, 
                            'acc': val_acc,
                            'best_acc': best_acc, 
                            'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "warmup_model.pth") )
    
    # ==== standard training ====
    # 添加双采样分支
      
    logger.info('== The class weight type of standard training is {}'.format(cfg.MODEL.LOSS.LABELED_LOSS_CLASS_WEIGHT_TYPE))
    
    start_epoch=warmup_epoch+1   
    
    feat,feat_y,feat_idx=prepare_feat(model, domain_trainloader,cfg=cfg) 
    feat_y=update_domain_label(feat,feat_y,cfg) 
        
    for epoch in range(start_epoch, max_epoch+1):
        start_time = time.time() 
        logger.info('== Train Epoch: [{}|{}]'.format(epoch, max_epoch))
        train_acc,train_group_acc,train_loss, train_loss_x, train_loss_u,\
        dl_center,dl_num_count,dl_conf_thres= \
            train(labeled_trainloader, unlabeled_trainloader, \
                  model, optimizer,  \
                  l_criterion,ul_criterion, \
                  epoch,logger,cfg,\
                  dl_center,dl_num_count,dl_conf_thres, \
                  dual_labeled_trainloader=dual_labeled_trainloader,feat=feat,feat_y=feat_y
       )
        # scheduler.step()  
        val_loss, val_acc,val_group_acc = validate(val_loader,model, val_criterion, epoch=epoch,cfg=cfg, mode='Valid Stats')
        test_loss, test_acc ,test_group_acc= validate(test_loader,model, val_criterion, epoch=epoch,cfg=cfg, mode='Test Stats')

        end_time = time.time()
        # loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        # acc        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        # group_acc
        train_group_accs.append(train_group_acc)
        test_group_accs.append(test_group_acc)
        val_group_accs.append(val_group_acc)
        # append logger file 
        logger.info('== Train_loss:{:>5.4f}  train_loss_x:{:>5.4f}   train_loss_u:{:>5.4f}  val_losss:{:>5.4f}   test_loss:{:>5.4f}   Epoch_Time:{:>5.2f}min'.\
                        format(train_loss, train_loss_x, train_loss_u, val_loss, test_loss,(end_time - start_time) / 60))
        logger.info('== Val  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(val_group_acc[0]*100,val_group_acc[1]*100,val_group_acc[2]*100))
        logger.info('== Test group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(test_group_acc[0]*100,test_group_acc[1]*100,test_group_acc[2]*100))
        logger.info('== Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(val_acc*100,test_acc*100))
        feat,domain_y,_=prepare_feat(model, domain_trainloader,feat=feat,feat_idx=feat_idx,cfg=cfg) 
        # if epoch % cfg.ALGORITHM.OOD_DETECTOR.DETECT_EPOCH==0:
        #     feat_y=update_domain_label(feat,feat_y,cfg) 
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
         
          
        logger.info('='*100)
         
    logger.info('Mean test acc:{:>5.2f}'.format(np.mean(test_accs[-20:]))) 
    # 画出训练过程中每个分组的acc变化折线图 
    plot_group_acc_over_epoch(group_acc=train_group_accs,title="Train Group Average Accuracy",save_path=os.path.join(pic_dir,'train_group_acc.jpg'),warmup_epoch=warmup_epoch)
    plot_group_acc_over_epoch(group_acc=val_group_accs,title="Val Group Average Accuracy",save_path=os.path.join(pic_dir,'val_group_acc.jpg'),warmup_epoch=warmup_epoch)
    plot_group_acc_over_epoch(group_acc=test_group_accs,title="Test Group Average Accuracy",save_path=os.path.join(pic_dir,'test_group_acc.jpg'),warmup_epoch=warmup_epoch)
    plot_acc_over_epoch(train_accs,title="Train average accuracy",save_path=os.path.join(pic_dir,'train_acc.jpg'),)
    plot_acc_over_epoch(test_accs,title="Test average accuracy",save_path=os.path.join(pic_dir,'test_acc.jpg'),)
    plot_acc_over_epoch(val_accs,title="Val average accuracy",save_path=os.path.join(pic_dir,'val_acc.jpg'),)
    plot_loss_over_epoch(train_losses,title="Train Average Loss",save_path=os.path.join(pic_dir,'train_loss.jpg'))
    plot_loss_over_epoch(val_losses,title="Val Average Loss",save_path=os.path.join(pic_dir,'val_loss.jpg'))
    plot_loss_over_epoch(test_losses,title="Test Average Loss",save_path=os.path.join(pic_dir,'test_loss.jpg'))
    
def load_warmup_model(model,cfg,optimizer):
    warmup_model_path=get_warmup_model_path(cfg)
    if not os.path.exists(warmup_model_path):
        return model,1,optimizer
    [model,start_epoch,optimizer_dict]=load_checkpoint(model, warmup_model_path,cfg=cfg,return_start_epoch=True,return_optimizer=True)
    optimizer.load_state_dict(optimizer_dict) 
    return  model,start_epoch+1,optimizer
    
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
            trainer=MOODTrainer(cfg)
            trainer.train()  
            # trainer.evaluate()         
            print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
    
    
    
    

