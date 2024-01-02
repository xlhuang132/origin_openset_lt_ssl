
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn 
import torch.backends.cudnn as cudnn    
import numpy as np 
import models 
import time  
import os    
import datetime
import copy
import torch.nn.functional as F
from dataset.build_dataloader import build_dual_l_loader,_build_loader
from utils.ema_model import EMAModel  
from .base_trainer import BaseTrainer
from utils import FusionMatrix,AverageMeter
from utils.build_optimizer import get_optimizer_params
from models.cross_model_matching_head import CrossModalMatchingHead
import math
from loss.contrastive_loss import * 
from torch.utils.data import DataLoader, Subset
from dataset.base import BaseNumpyDataset 
from skimage.filters import threshold_otsu
from dataset.build_dataset import merge_dataset
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class T2TTrainer(BaseTrainer):
    def __init__(self, cfg):        
        super().__init__(cfg)      
        self.num_workers=self.cfg.DATASET.NUM_WORKERS
        # dataset
        l_dataset=self.labeled_trainloader.dataset 
        self.init_l_data, self.l_transforms = l_dataset.select_dataset(return_transforms=True) 
        ul_dataset=self.unlabeled_trainloader.dataset 
        self.init_ul_data, self.ul_transforms = ul_dataset.select_dataset(return_transforms=True)         
        self.udst_rotnet=BaseNumpyDataset(merge_dataset(copy.deepcopy(self.init_l_data),copy.deepcopy(self.init_ul_data)), transforms=self.l_transforms,num_classes=self.num_classes)
        self.udst_rotnet_loader =_build_loader(self.cfg, self.udst_rotnet,is_train=False)
        self.rotnet_iter=iter(self.udst_rotnet_loader) 
        
        _,self.test_transform=self.test_loader.dataset.select_dataset(return_transforms=True)
        self.udst_eval_dataset=BaseNumpyDataset(copy.deepcopy(self.init_ul_data), transforms=self.test_transform,num_classes=self.num_classes)
        self.udst_eval_loader=_build_loader(self.cfg, self.udst_eval_dataset,is_train=False)
        
        # model
        self.cmm_head = CrossModalMatchingHead(self.num_classes, 512).cuda()
        self.rotnet_head=torch.nn.Linear(512, 4).cuda()  
        # hyper-params
        self.stage1_steps=self.cfg.ALGORITHM.T2T.STAGE1_STEPS
        self.stage2_steps=self.cfg.ALGORITHM.T2T.STAGE2_STEPS
        self.filter_every_step=self.cfg.ALGORITHM.T2T.FILTER_EVERY_STEP
        self.mu=self.cfg.ALGORITHM.T2T.MU
        self._rebuild_optimizer()
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)
        
    def loss_init(self):       
         
        # In stage1, we train the model with three losses:
        # 1. Lx:  cross-entropy loss for labeled data (ref to L_ce in origin paper)
        # 2. Lmx: cross-modal matching loss for labeled data (ref to L_cm^l in origin paper)
        # 3. Lr:  rotation recognition loss for all training data (ref to L_rot in origin paper)
        
        """
        In this stage, we train the model with five losses:
        1. Lx:  cross-entropy loss for labeled data (ref to L_ce)
        2. Lmx: cross-modal matching loss for labeled data (ref to L_cm^l)
        3. Lr:  rotation recognition loss for all training data (ref to L_rot)
        4. Lmu: cross-modal matching loss for unlabeled data (ref to L_cm^u)
        5. Lu:  consistency constraint loss for filtered unlabeled data (ref to L_cc)
        """
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_mx = AverageMeter()
        self.losses_r = AverageMeter()  
        self.losses_mu = AverageMeter()
        self.losses_u = AverageMeter()  
    
    def train(self,):
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()      
        self.loss_init()
        start_time = time.time()   
        for self.iter in range(self.start_iter, self.max_iter):             
            self.epoch= (self.iter // self.val_iter)+1  
            return_data=self.train_step()
            if return_data is not None:
                pred,gt=return_data[0],return_data[1]
                fusion_matrix.update(pred, gt) 
            if self.iter%self.val_iter==0:  
                end_time = time.time()           
                time_second=(end_time - start_time)
                eta_seconds = time_second * (self.max_epoch - self.epoch)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS)
                self.train_group_accs.append(group_acc)
                results=self.evaluate()
                
                if self.best_val<results[0]:
                    self.best_val=results[0]
                    self.best_val_test=results[1]
                    self.best_val_iter=self.iter
                    self.save_checkpoint(file_name="best_model.pth")
                if self.epoch%self.save_epoch==0:
                    self.save_checkpoint()
                
                self.logger.info("== Pretraining is enable:{}".format(self.pretraining))
                self.logger.info('== Train_loss:{:>5.4f}  train_loss_x:{:>5.4f}   '.\
                    format(self.losses.avg, self.losses_x.avg))
                self.logger.info('== val_losss:{:>5.4f}   test_loss:{:>5.4f}   epoch_Time:{:>5.2f}min eta:{}'.\
                        format(self.val_losses[-1], self.test_losses[-1],time_second / 60,eta_string))
                self.logger.info('== Train  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.train_group_accs[-1][0]*100,self.train_group_accs[-1][1]*100,self.train_group_accs[-1][2]*100))
                self.logger.info('==  Val   group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.val_group_accs[-1][0]*100,self.val_group_accs[-1][1]*100,self.val_group_accs[-1][2]*100))
                self.logger.info('==  Test  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.test_group_accs[-1][0]*100,self.test_group_accs[-1][1]*100,self.test_group_accs[-1][2]*100))
                self.logger.info('== Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(results[0]*100,results[1]*100))
                self.logger.info('== Best Results: Epoch:{} Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(self.best_val_iter//self.val_iter,self.best_val*100,self.best_val_test*100))
              
                # reset 
                fusion_matrix = FusionMatrix(self.num_classes)
                acc = AverageMeter()                 
                self.epoch= (self.iter // self.val_iter)+1                
                self.loss_init() 
                start_time = time.time()    
                self.operate_after_epoch()
                
                
        self.plot()       
        return
    

    def train_step(self,pretraining=False):
        if 0<=self.iter<=self.stage1_steps:
            return self.train_stage_1_step()
        else:
            return self.train_stage_2_step()
        
        
    def train_stage_1_step(self): 
        self.model.train()
        self.cmm_head.train()
        self.rotnet_head.train()
        loss =0
        # DL  
        try:        
            inputs_x, targets_x,index_x = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,index_x = self.labeled_train_iter.next() 
        
        batch_size = inputs_x.shape[0]
        # DU   
        try:
            (inputs_u_w, _), gt_u, index_u = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter = iter(self.unlabeled_trainloader)
            (inputs_u_w, _), gt_u, index_u = self.unlabeled_train_iter.next()
        
        
        # rotate unlabeled data with 0, 90, 180, 270 degrees
        inputs_r = torch.cat(
            [torch.rot90(inputs_u_w, i, [2, 3]) for i in range(4)], dim=0)
        targets_r = torch.cat(
            [torch.empty(inputs_u_w.size(0)).fill_(i).long() for i in range(4)], dim=0).cuda()
    
        inputs_x=inputs_x.cuda()
        targets_x=targets_x.cuda()
         
        feats_x = self.model(inputs_x,return_encoding=True)
        logits_x = self.model(feats_x,classifier=True)
       
        # Cross Entropy Loss for Labeled Data
        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)         
        # Cross Modal Matching Training: 1 positve pair + 2 negative pair for each labeled data
        # [--pos--, --hard_neg--, --easy_neg--]
        matching_gt = torch.zeros(3 * batch_size).cuda()
        matching_gt[:batch_size] = 1
        y_onehot = torch.zeros((3 * batch_size, self.num_classes)).float().cuda()
        y = torch.zeros(3 * batch_size).long().cuda()
        y[:batch_size] = targets_x
        with torch.no_grad():
            prob_sorted_index = torch.argsort(logits_x, descending=True)
            for i in range(batch_size):
                if prob_sorted_index[i, 0] == targets_x[i]:
                    y[1 * batch_size + i] = prob_sorted_index[i, 1]
                    y[2 * batch_size + i] = int(np.random.choice(prob_sorted_index[i, 2:].cpu(), 1))
                else:
                    y[1 * batch_size + i] = prob_sorted_index[i, 0]
                    choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                    while choice == targets_x[i]:
                        choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                    y[2 * batch_size + i] = choice
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        matching_score_x = self.cmm_head(feats_x.repeat(3, 1), y_onehot)
        Lmx = F.binary_cross_entropy_with_logits(matching_score_x.view(-1), matching_gt)

        # Cross Entropy Loss for Rotation Recognition
        inputs_r = inputs_r.cuda()
        feats_r = self.model(inputs_r, return_encoding=True)
        Lr = F.cross_entropy(self.rotnet_head(feats_r), targets_r, reduction='mean')

        loss = Lx + Lmx + Lr
 
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0))
        self.losses_mx.update(Lmx.item(), inputs_x.size(0)) 
        self.losses_r.update(Lr.item(), inputs_u_w.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Stage 1 Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_mx:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_mx.val))
            self.logger.info('========== Loss_r:{:>5.4f}  ==========='.format(self.losses_r.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
    
    def train_stage_2_step(self):  
        # clean unlabeled data periodically for consistency constraint loss
        if (self.iter-1) % self.filter_every_step == 0:
            in_dist_idxs = self.filter_ood(self.cfg, self.udst_eval_loader, self.model, self.cmm_head)
            in_dist_unlabeled_np={"images":self.udst_eval_dataset.dataset["images"][in_dist_idxs],
                                  "labels":self.udst_eval_dataset.dataset["labels"][in_dist_idxs]}
            in_dist_unlabeled_dataset = BaseNumpyDataset(copy.deepcopy(in_dist_unlabeled_np), transforms=self.ul_transforms,num_classes=self.num_classes)
            # Subset(self.unlabeled_trainloader.dataset, in_dist_idxs)
            self.unlabeled_trainloader = DataLoader(
                in_dist_unlabeled_dataset,
                batch_size=self.batch_size*self.mu,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
        self.model.train() 
        self.cmm_head.train() 
        self.rotnet_head.train()
        loss=0
        try:
            inputs_x, targets_x, index_x =  self.labeled_train_iter.next()
        except:
            self.labeled_train_iter = iter(self.labeled_trainloader)
            inputs_x, targets_x, index_x =  self.labeled_train_iter.next()

        try:
            data = self.unlabeled_train_iter.next()
            (inputs_u_w, inputs_u_s), gt_u, index_u=data
        except:
            self.unlabeled_train_iter = iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next()
            (inputs_u_w, inputs_u_s), gt_u, index_u=data
            # (inputs_u_w, inputs_u_s), gt_u, index_u = self.unlabeled_train_iter.next()
        
        try:
            inputs_r, gt_u, index_u = self.rotnet_iter.next()
        except:
            self.rotnet_iter = iter(self.udst_rotnet_loader)
            inputs_r, gt_u, index_u = self.rotnet_iter.next()
        # rotate unlabeled data with 0, 90, 180, 270 degrees
        inputs_r = torch.cat(
            [torch.rot90(inputs_r, i, [2, 3]) for i in range(4)], dim=0)
        targets_r = torch.cat(
            [torch.empty(index_u.size(0)).fill_(i).long() for i in range(4)], dim=0).cuda()
 
        
        batch_size = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).cuda()
        targets_x = targets_x.cuda()

        feats = self.model(inputs, return_encoding=True)
        logits = self.model(feats,classifier=True)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        feats_x = feats[:batch_size]
        # del logits

        # Cross Entropy Loss for Labeled Data
        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)   
        # Consistency Constraint Loss for Unlabeled Data
        # hyper parameters for UDA
        T = 0.4
        p_cutoff = 0.8
        logits_tgt = logits_u_w / T
        probs_u_w = torch.softmax(logits_u_w, dim=1)
        loss_mask = probs_u_w.max(-1)[0].ge(p_cutoff)

        if loss_mask.sum() == 0:
            Lu = torch.zeros(1, dtype=torch.float).cuda()
        else:
            Lu = F.kl_div(
                torch.log_softmax(logits_u_s[loss_mask], -1), 
                torch.softmax(logits_tgt[loss_mask].detach().data, -1),
                reduction='batchmean')

        # Cross Modal Matching Training:
        # 1 positve pair + 2 negative pairs for each labeled data
        # [--pos--, --hard_neg--, --easy_neg--]
        matching_gt = torch.zeros(3 * batch_size).cuda()
        matching_gt[:batch_size] = 1
        y_onehot = torch.zeros((3 * batch_size, self.num_classes)).float().cuda()
        y = torch.zeros(3 * batch_size).long().cuda()
        y[:batch_size] = targets_x
        with torch.no_grad():
            prob_sorted_index = torch.argsort(logits_x, descending=True)
            for i in range(batch_size):
                if prob_sorted_index[i, 0] == targets_x[i]:
                    y[1 * batch_size + i] = prob_sorted_index[i, 1]
                    y[2 * batch_size + i] = int(np.random.choice(prob_sorted_index[i, 2:].cpu(), 1))
                else:
                    y[1 * batch_size + i] = prob_sorted_index[i, 0]
                    choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                    while choice == targets_x[i]:
                        choice = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
                    y[2 * batch_size + i] = choice
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        matching_score_x = self.cmm_head(feats_x.repeat(3, 1), y_onehot)
        Lmx = F.binary_cross_entropy_with_logits(matching_score_x.view(-1), matching_gt)

        # Cross Entropy Loss for Rotation Recognition
        inputs_r = inputs_r.cuda()
        feats_r = self.model(inputs_r, return_encoding=True)
        logits_r=self.model(feats_r,classifier=True)
        Lr = F.cross_entropy(self.rotnet_head(feats_r), targets_r, reduction='mean')

        # Cross Modal Matching Training:
        # Use Entropy Minimization Loss for all unlabeled data (including OOD data)
        # So we use data from RotNet Dataloder which has all training data
        batch_size = inputs_r.size(0) // 4
        y_onehot = torch.zeros((2 * batch_size, self.num_classes)).float().cuda()
        y = torch.zeros(2 * batch_size).long().cuda()
        # select the most confident class and randomly choose one from rest classes
        with torch.no_grad():
            prob_sorted_index = torch.argsort(logits_r[:batch_size], descending=True)
            y[:batch_size] = prob_sorted_index[:, 0]
            for i in range(batch_size):
                y[batch_size + i] = int(np.random.choice(prob_sorted_index[i, 1:].cpu(), 1))
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        matching_score_u = self.cmm_head(feats_r[:batch_size].repeat(2, 1), y_onehot)
        Lmu = F.binary_cross_entropy_with_logits(matching_score_u, torch.sigmoid(matching_score_u))

        # we use linear ramp up weighting here for stabilizing training process
        alpha = linear_rampup(max(0,self.iter-self.stage1_steps), self.stage2_steps)
        loss = Lx + Lmx + Lr + alpha * (Lmu + Lu)
        
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0))
        self.losses_mx.update(Lmx.item(), inputs_x.size(0)) 
        self.losses_r.update(Lr.item(), inputs_u_s.size(0))
        self.losses_mu.update(Lmu.item(), inputs_u_s.size(0)) 
        self.losses_u.update(Lu.item(), inputs_u_s.size(0))

         # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('==  Stage 2 Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_mx:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_mx.val))
            self.logger.info('========== Loss_r:{:>5.4f} Loss_mu:{:>5.4f} Loss_u:{:>5.4f} ==========='.format(self.losses_r.val,self.losses_mu.val,self.losses_u.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()

      
    def filter_ood(self,cfg, loader, model, cmm_head):
        # switch to evaluate mode
        model.eval()
        cmm_head.eval()
        matching_scores = []
        targets = []
        idxs = []
        in_dist_idxs = []
        ood_cnt = 0

        with torch.no_grad():
            for batch_idx, (input, target, indexs) in enumerate(loader):
                input = input.cuda()
                feats = model(input, return_encoding=True)
                logits = model(feats,classifier=True)
                y_onehot = torch.zeros((input.size(0), self.num_classes)).float().cuda()
                y_pred = torch.argmax(logits, dim=1, keepdim=True)
                y_onehot.scatter_(1, y_pred, 1)

                matching_score = torch.sigmoid(cmm_head(feats, y_onehot))

                for i in range(len(target)):
                    matching_scores.append(matching_score[i].cpu().item())
                    idxs.append(indexs[i].item())
                    targets.append(target[i].item())

        # use otsu threshold to adaptively compute threshold
        matching_scores = np.array(matching_scores)
        thresh = threshold_otsu(matching_scores)
        for i, s in enumerate(matching_scores):
            if s > thresh:
                in_dist_idxs.append(idxs[i])
                if targets[i] == -1:
                    ood_cnt += 1
        self.logger.info('OOD Filtering threshold: %.3f' % thresh)
        self.logger.info('false positive: %d/%d' % (ood_cnt, len(in_dist_idxs)))
        # switch back to train mode
        model.train()
        cmm_head.train()
        return in_dist_idxs
    
    def _rebuild_optimizer(self):
        grouped_parameters = [
        {'params': self.model.parameters()},
        {'params': self.rotnet_head.parameters()},
        {'params': self.cmm_head.parameters()}
        ]

        self.optimizer = get_optimizer_params(self.cfg,params=grouped_parameters)
         