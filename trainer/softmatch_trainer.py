
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
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
from loss.consistency_loss import consistency_loss
from loss.cross_entropy import ce_loss
import contextlib
from utils import FusionMatrix

from .hooks import *
from collections import OrderedDict 

class SoftMatchTrainer(BaseTrainer):
    def __init__(self, cfg):        
        super().__init__(cfg)      
        
        self.T = self.cfg.ALGORITHM.SOFTMATCH.T
        self.use_hard_label = self.cfg.ALGORITHM.SOFTMATCH.HARD_LABEL
        self.dist_align = self.cfg.ALGORITHM.SOFTMATCH.DIST_ALIGN
        self.dist_uniform = self.cfg.ALGORITHM.SOFTMATCH.DIST_UNIFORM
        self.ema_p = self.cfg.ALGORITHM.SOFTMATCH.EMA_P
        self.n_sigma = self.cfg.ALGORITHM.SOFTMATCH.N_SIGMA
        self.per_class = self.cfg.ALGORITHM.SOFTMATCH.PER_CLASS 
        self.use_amp = self.cfg.ALGORITHM.SOFTMATCH.AMP
        self.lambda_u=self.cfg.ALGORITHM.SOFTMATCH.LAMBDA_U
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.use_cat=True 
        self.ce_loss = ce_loss
        self.consistency_loss = consistency_loss   
        self.distributed=False
        self.world_size=1     
        self._hooks = []  # record underlying hooks
        self.hooks_dict = OrderedDict()  # actual object to be used to call hooks
        self.set_hooks()
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)
        
        
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.ema_p, p_target_type='uniform' if self.dist_uniform else 'model'), 
            "DistAlignHook")
        self.register_hook(SoftMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.n_sigma, momentum=self.ema_p, per_class=self.per_class), "MaskingHook")
        # super().set_hooks()    
        
    def register_hook(self, hook, name=None, priority="NORMAL"):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263  # noqa: E501
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered.
                Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", used to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

        
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
            (inputs_u_w, inputs_u_s), gt_u, index_u = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter = iter(self.unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), gt_u, index_u = self.unlabeled_train_iter.next()
        
        
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)        
        inputs_u_w , inputs_u_s= inputs_u_w.cuda(),inputs_u_s.cuda()   
        num_lb = inputs_x.shape[0]
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
                outputs_feats = self.model(inputs,return_encoding=True)
                outputs_logits = self.model(outputs_feats,classifier=True)
                logits_x_lb = outputs_logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs_logits[num_lb:].chunk(2)
                feats_x_lb = outputs_feats[:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs_feats[num_lb:].chunk(2)
            else:                 
                feats_x_lb = self.model(inputs_x,return_encoding=True) 
                logits_x_lb = self.model(feats_x_lb,classifier=True)  
                feats_x_ulb_s = self.model(inputs_u_s,return_encoding=True)
                logits_x_ulb_s = self.model(feats_x_ulb_s,classifier=True)
                with torch.no_grad(): 
                    feats_x_ulb_w = self.model(inputs_u_w,return_encoding=True) 
                    logits_x_ulb_w = self.model(feats_x_ulb_w,classifier=True)
            feat_dict = {'inputs_x':feats_x_lb, 'inputs_u_w':feats_x_ulb_w, 'inputs_u_s':feats_x_ulb_s}


            sup_loss = self.ce_loss(logits_x_lb, targets_x, reduction='mean')

            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # uniform distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          # make sure this is logits, not dist aligned probs
                                          # uniform alignment in softmatch do not use aligned probs for generating pseudo labels
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

            # calculate loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            loss = sup_loss + self.lambda_u * unsup_loss
        # compute 1st branch accuracy
        score_result = self.func(logits_x_lb)
        now_result = torch.argmax(score_result, 1)         
         
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(sup_loss.item(), inputs_x.size(0))
        self.losses_u.update(unsup_loss.item(), inputs_u_w.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_u.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
     

