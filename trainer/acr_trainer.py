
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters,AverageMeter
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np 
import models 
import time 
import torch.optim as optim 
import os   
import datetime
import torch.nn.functional as F  
from .base_trainer import BaseTrainer

from utils import FusionMatrix

def compute_py(train_loader, cfg):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell,_) in enumerate(train_loader):
        labell = labell.cuda()
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.cuda()
    return label_freq_array


def compute_adjustment(train_loader, tro, cfg):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, labell,_) in enumerate(train_loader):
        labell = labell.cuda()
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.cuda()
    return adjustments


def compute_adjustment_by_py(py, tro, cfg):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.cuda()
    return adjustments


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class ACRTrainer(BaseTrainer):  
    def __init__(self, cfg):        
        super().__init__(cfg) 
        
        self.mask_probs = AverageMeter() 
        self.est_epoch=self.cfg.ALGORITHM.ACR.EST_EPOCH
        self.est_step=0
        
        self.py_con = compute_py(self.labeled_trainloader, cfg)
        self.py_uni = torch.ones(self.num_classes) / self.num_classes
        self.py_rev = torch.flip(self.py_con, dims=[0])        
        self.py_uni = self.py_uni.cuda()

        self.tau1=self.cfg.ALGORITHM.ACR.TAU1
        self.tau12=self.cfg.ALGORITHM.ACR.TAU12
        self.tau2=self.cfg.ALGORITHM.ACR.TAU2
        self.taumin=0
        self.taumax=self.tau1
        
        self.adjustment_l1 = compute_adjustment_by_py(self.py_con, self.tau1, cfg)
        self.adjustment_l12 = compute_adjustment_by_py(self.py_con, self.tau12, cfg)
        self.adjustment_l2 = compute_adjustment_by_py(self.py_con, self.tau2, cfg)
        self.T=self.cfg.ALGORITHM.ACR.T
        self.mu=self.cfg.ALGORITHM.ACR.MU
        self.threshold=self.cfg.ALGORITHM.ACR.THRESHOLD
        self.ema_u =self.cfg.ALGORITHM.ACR.EMA_U   
        
        self.u_py = torch.ones(self.num_classes) / self.num_classes
        self.u_py = self.u_py.cuda()     
        self.KL_div = nn.KLDivLoss(reduction='sum')
    
     

    def train(self,):
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()       
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter()  
        self.mask_probs = AverageMeter() 
        self.epoch= (self.iter // self.val_iter)+1  
        start_time = time.time()   
        for self.iter in range(self.start_iter, self.max_iter):
            if self.epoch > self.est_epoch:
                self.count_KL = self.count_KL / self.val_iter
                KL_softmax = (torch.exp(self.count_KL[0])) / (torch.exp(self.count_KL[0])+torch.exp(self.count_KL[1])+torch.exp(self.count_KL[2]))
                tau = self.taumin + (self.taumax - self.taumin) * KL_softmax
                if math.isnan(tau)==False:
                    self.adjustment_l1 = compute_adjustment_by_py(self.py_con, tau, self.cfg)

            self.count_KL = torch.zeros(3).cuda()
            
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
                self.logger.info('== Train_loss:{:>5.4f}  train_loss_x:{:>5.4f}   train_loss_u:{:>5.4f} '.\
                    format(self.losses.avg, self.losses_x.avg, self.losses_u.avg))
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
                self.losses = AverageMeter()
                self.losses_x = AverageMeter()
                self.losses_u = AverageMeter()  
                self.mask_probs = AverageMeter()                
                start_time = time.time()    
                
                
        self.plot()       
        return
    
    
    def train_step(self,pretraining=False):
        self.model.train()
        loss =0
        # DL  
        try:        
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        
        # DU   
        try:       
            data = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next() 
        inputs_u_w, inputs_u_s, inputs_u_s1, u_real = data[0]
         
        u_real = u_real.cuda()
        mask_l = (u_real != -2)
        mask_l = mask_l.cuda()

             
        batch_size = inputs_x.shape[0]
        inputs = interleave(
            torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3*self.mu+1).cuda()
        targets_x = targets_x.cuda()

        logits_feat = self.model(inputs,return_encoding=True)
        logits = self.model(logits_feat,classifier=True)

        logits = de_interleave(logits, 3*self.mu+1)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s, logits_u_s1 = logits[batch_size:].chunk(3)
        del logits
        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        logits_b = self.model(logits_feat,classifier1=True)

        logits_b = de_interleave(logits_b, 3 * self.mu + 1)
        logits_x_b = logits_b[:batch_size]
        logits_u_w_b, logits_u_s_b, logits_u_s1_b = logits_b[batch_size:].chunk(3)
        del logits_b
        Lx_b = F.cross_entropy(logits_x_b + self.adjustment_l2, targets_x, reduction='mean')

        pseudo_label = torch.softmax((logits_u_w.detach() - self.adjustment_l1) / self.T, dim=-1)
        pseudo_label_h2 = torch.softmax((logits_u_w.detach() - self.adjustment_l12) / self.T, dim=-1)
        pseudo_label_b = torch.softmax(logits_u_w_b.detach() / self.T, dim=-1)
        pseudo_label_t = torch.softmax(logits_u_w.detach() / self.T, dim=-1)

        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)
        max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)
        max_probs_t, targets_u_t = torch.max(pseudo_label_t, dim=-1)

        mask = max_probs.ge(self.threshold)
        mask_h2 = max_probs_h2.ge(self.threshold)
        mask_b = max_probs_b.ge(self.threshold)
        mask_t = max_probs_t.ge(self.threshold)

        mask_ss_b_h2 = mask_b + mask_h2
        mask_ss_t = mask + mask_t

        mask = mask.float()
        mask_b = mask_b.float()

        mask_ss_b_h2 = mask_ss_b_h2.float()
        mask_ss_t = mask_ss_t.float()

        mask_twice_ss_b_h2 = torch.cat([mask_ss_b_h2, mask_ss_b_h2], dim=0).cuda()
        mask_twice_ss_t = torch.cat([mask_ss_t, mask_ss_t], dim=0).cuda()

        logits_u_s_twice = torch.cat([logits_u_s, logits_u_s1], dim=0).cuda()
        targets_u_twice = torch.cat([targets_u, targets_u], dim=0).cuda()
        targets_u_h2_twice = torch.cat([targets_u_h2, targets_u_h2], dim=0).cuda()

        logits_u_s_b_twice = torch.cat([logits_u_s_b, logits_u_s1_b], dim=0).cuda()

        now_mask = torch.zeros(self.num_classes)
        now_mask = now_mask.cuda()
        u_real[u_real==-2] = 0

        if self.epoch > self.est_epoch:
            now_mask[targets_u_b] += mask_l*mask_b
            self.est_step = self.est_step + 1

            if now_mask.sum() > 0:
                now_mask = now_mask / now_mask.sum()
                self.u_py = self.ema_u * self.u_py + (1-self.ema_u) * now_mask
                KL_con = 0.5 * self.KL_div(self.py_con.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_con)
                KL_uni = 0.5 * self.KL_div(self.py_uni.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_uni)
                KL_rev = 0.5 * self.KL_div(self.py_rev.log(), self.u_py) + 0.5 * self.KL_div(self.u_py.log(), self.py_rev)
                self.count_KL[0] = self.count_KL[0] + KL_con
                self.count_KL[1] = self.count_KL[1] + KL_uni
                self.count_KL[2] = self.count_KL[2] + KL_rev
#==================================
        Lu = (F.cross_entropy(logits_u_s_twice, targets_u_twice,
                                reduction='none') * mask_twice_ss_t).mean()
        Lu_b = (F.cross_entropy(logits_u_s_b_twice, targets_u_h2_twice,
                                reduction='none') * mask_twice_ss_b_h2).mean()
        loss = Lx + Lu + Lx_b + Lu_b 
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()      
        self.mask_probs.update(mask.mean().item())
        self.losses.update(loss.item())
        self.losses_x.update(Lx.item()+Lx_b.item())
        self.losses_u.update(Lu.item()+Lu_b.item())
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_u.val))
        
        return
     