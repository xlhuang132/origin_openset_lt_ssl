
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
import faiss
import datetime
import torch.nn.functional as F
from dataset.build_dataloader import build_dual_l_loader,_build_loader
from utils.ema_model import EMAModel  
from .base_trainer import BaseTrainer
from utils import FusionMatrix,AverageMeter
from models.projector import  Projector
import math
from loss.contrastive_loss import *
from dataset.base import BaseNumpyDataset
from models.feature_queue import FeatureQueue
class MOODTrainer(BaseTrainer):
    def __init__(self, cfg):        
        super().__init__(cfg)      
        
        self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True)  
        # self.projector=Projector(cfg).cuda()        
        self.l_num=len(self.labeled_trainloader.dataset)
        self.ul_num=len(self.unlabeled_trainloader.dataset) 
        self.id_masks=torch.ones(self.ul_num).cuda()
        self.ood_masks=torch.zeros(self.ul_num).cuda()
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)        
        self.warmup_temperature=self.cfg.ALGORITHM.PRE_TRAIN.SimCLR.TEMPERATURE
   
        l_dataset=self.labeled_trainloader.dataset 
        self.init_l_data, self.l_transforms = l_dataset.select_dataset(return_transforms=True)
        
        ul_dataset=self.unlabeled_trainloader.dataset 
        self.init_ul_data, self.ul_transforms = ul_dataset.select_dataset(return_transforms=True)
        self.warmup_iter=cfg.ALGORITHM.MOOD.WARMUP_ITER 
        
        # ========= build dual loader ====================
        mood_total_samples=self.batch_size*(self.max_iter-self.warmup_iter)
        self.dual_l_loader=build_dual_l_loader(self.init_l_data,self.l_transforms,cfg=self.cfg,total_samples=mood_total_samples,sampler_name="ReversedSampler")
        self.dual_l_iter=iter(self.dual_l_loader)       
        # =========== for ood detector ==============
        self.feature_dim=cfg.ALGORITHM.MOOD.OOD_DETECTOR.FEATURE_DIM 
        self.feature_decay=0.9
        self.k=cfg.ALGORITHM.MOOD.OOD_DETECTOR.K    
        self.update_domain_y_iter=cfg.ALGORITHM.MOOD.OOD_DETECTOR.DOMAIN_Y_UPDATE_ITER
        self.update_OOD_DETECTOR_iter=cfg.ALGORITHM.MOOD.OOD_DETECTOR.OOD_DETECTOR_UPDATE_ITER
        self.lambda_pap=cfg.ALGORITHM.MOOD.PAP_LOSS_WEIGHT
        self.feature_loss_temperature=cfg.ALGORITHM.MOOD.FEATURE_LOSS_TEMPERATURE
 
        self.alpha = cfg.ALGORITHM.MOOD.MIXUP_ALPHA 
        
        self.ood_detect_fusion = FusionMatrix(2)   
        self.id_detect_fusion = FusionMatrix(2)  
        
        self.ablation_enable=cfg.ALGORITHM.ABLATION.ENABLE
        self.dual_branch_enable=cfg.ALGORITHM.ABLATION.DUAL_BRANCH
        self.mixup_enable=cfg.ALGORITHM.ABLATION.MIXUP
        self.ood_detection_enable=cfg.ALGORITHM.ABLATION.OOD_DETECTION
        self.pap_loss_enable=cfg.ALGORITHM.ABLATION.PAP_LOSS 
        self.pap_loss_weight=cfg.ALGORITHM.MOOD.PAP_LOSS_WEIGHT
        
        #2023.11.7
        l_dataset = self.labeled_trainloader.dataset 
        l_data_np,_ = l_dataset.select_dataset(return_transforms=True)
        _,trans=self.test_loader.dataset.select_dataset(return_transforms=True)
        new_l_dataset = BaseNumpyDataset(l_data_np, transforms=trans,num_classes=self.num_classes)
        self.test_labeled_trainloader = _build_loader(self.cfg, new_l_dataset,is_train=False)
        ul_dataset = self.unlabeled_trainloader.dataset 
        ul_data_np,_ = ul_dataset.select_dataset(return_transforms=True)
        new_ul_dataset = BaseNumpyDataset(ul_data_np, transforms=trans,num_classes=self.num_classes)
        self.test_unlabeled_trainloader = _build_loader(self.cfg, new_ul_dataset,is_train=False)
        self.loss_init()
        
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_pap = AverageMeter() 
        self.losses_pap_id = AverageMeter() 
        self.losses_pap_ood = AverageMeter() 

    def train_step(self,pretraining=False):
        if self.pretraining:
            return self.train_warmup_step()
        else:
            return self.train_mood_step()
    
    def train_warmup_step(self):
        self.model.train()
        loss =0
        # DL  
        try:
            (inputs_x,inputs_x2), _,_ = self.pre_train_iter.next()             
            # (inputs_x,inputs_x2), _ = self.pre_train_iter.next() 
        except:            
            self.pre_train_iter=iter(self.pre_train_loader)            
            (inputs_x,inputs_x2),_,_ = self.pre_train_iter.next() 
        # DU  
        # (inputs_u,inputs_u2),ul_y,_ = self.unlabeled_train_iter.next()
         
        inputs_x, inputs_x2 = inputs_x.cuda(), inputs_x2.cuda()
        # inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        
        # weak=torch.cat((inputs_x,inputs_u),0)
        # strong=torch.cat((inputs_x2,inputs_u2),0)
        # out_1 = self.model(weak,return_encoding=True) 
        # out_2 = self.model(strong,return_encoding=True) 
        
        out_1 = self.model(inputs_x,return_encoding=True) #torch.Size([64, 3, 32, 32])
        out_2 = self.model(inputs_x2,return_encoding=True) 
        # out_1=self.projector(out_1)        
        # out_2=self.projector(out_2)
        out_1=self.model.project_feature(out_1)        
        out_2=self.model.project_feature(out_2)  
        # with torch.no_grad():  
        #     self.queue.enqueue(out_1[:targets_x.size(0)].clone().detach(), targets_x.clone().detach())
        similarity  = pairwise_similarity(out_1,out_2,temperature=self.warmup_temperature) 
        loss        = NT_xent(similarity) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_u.val))
        return 

    def train_mood_step(self):
        self.model.train()
        loss_dict={}
        # DL  
        try:
            # (inputs_x,_), targets_x,_ = self.labeled_train_iter.next()
            inputs_x, targets_x,_ = self.labeled_train_iter.next()
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            # (inputs_x,_), targets_x,_ = self.labeled_train_iter.next()
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        # DU  
        try:
            (inputs_u,inputs_u2),ul_y,u_index = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            (inputs_u,inputs_u2),ul_y,u_index = self.unlabeled_train_iter.next()
            
            
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        ul_y=ul_y.cuda()            
        u_index=u_index.cuda()   
        # if torch.where(ul_y==-1).any():
        #     print()
        id_mask,ood_mask = torch.ones_like(ul_y).cuda(),torch.zeros_like(ul_y).cuda()         
        
        if not self.ablation_enable or self.ablation_enable and self.dual_branch_enable:            
            try:
                # (inputs_dual_x,_), targets_dual_x,_=self.dual_l_iter.next()                
                inputs_dual_x, targets_dual_x,_=self.dual_l_iter.next()
            except:
                self.dual_l_iter=iter(self.dual_labeled_trainloader)                 
                inputs_dual_x, targets_dual_x,_=self.dual_l_iter.next()           
                # (inputs_dual_x,_), targets_dual_x,_=self.dual_l_iter.next()
            inputs_dual_x, targets_dual_x = inputs_dual_x.cuda(), targets_dual_x.cuda(non_blocking=True)        
            
            if not self.ablation_enable or self.ablation_enable and self.mixup_enable:
                # mixup use ood
                if not self.ablation_enable or self.ablation_enable and self.ood_detection_enable:
                    id_mask=self.id_masks[u_index].detach()  
                    ood_mask=self.ood_masks[u_index].detach()   
                    ood_index = torch.nonzero(ood_mask, as_tuple=False).squeeze(1)  
                    inputs_dual_x=self.mix_up(inputs_dual_x, inputs_u[ood_index])
                else:                    
                    inputs_dual_x=self.mix_up(inputs_dual_x, inputs_u)
                # concat dl 
            inputs_x=torch.cat([inputs_x,inputs_dual_x],0)
            targets_x=torch.cat([targets_x,targets_dual_x],0)
        
        # 1. cls loss
        l_feature = self.model(inputs_x,return_encoding=True)  
        l_logits = self.model(l_feature,classifier=True)
        # 1. dl ce loss
        cls_loss = self.l_criterion(l_logits, targets_x)
        # loss_dict.update({"loss_cls": cls_loss})
        # compute 1st branch accuracy
        score_result = self.func(l_logits)
        now_result = torch.argmax(score_result, 1)          
        if torch.isnan(cls_loss).any():  
             print()
             
        # 2. cons loss
        ul_images=torch.cat([inputs_u , inputs_u2],0)
        ul_feature=self.model(ul_images,return_encoding=True) 
        ul_logits = self.model(ul_feature,classifier=True) 
        logits_weak, logits_strong = ul_logits.chunk(2)
        with torch.no_grad(): 
            p = logits_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1)

        loss_weight = confidence.ge(self.conf_thres).float()
        
        loss_weight*=id_mask
        cons_loss = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=logits_weak.size(0)
        )
        # loss_dict.update({"loss_cons": cons_loss})
        
        # l_feature2=self.projector(l_feature)
        l_feature2=self.model.project_feature(l_feature)
        with torch.no_grad():     
            self.queue.enqueue(l_feature2.clone().detach(), targets_x.clone().detach())
        
        pap_loss=torch.tensor(0.0).cuda()
        # 3. pap loss
        if not self.ablation_enable or self.ablation_enable and self.pap_loss_enable:
            # ul_feature2=self.projector(ul_feature)     
            ul_feature2=self.model.project_feature(ul_feature)
            ul_feature_weak,ul_feature_strong=ul_feature2.chunk(2)
            all_features=torch.cat((l_feature2,ul_feature_weak),dim=0)
            all_target=torch.cat((targets_x,pred_class),dim=0)
            confidenced_id_mask= torch.cat([torch.ones(l_feature2.size(0)).cuda(),id_mask*loss_weight],dim=0).long() 
             
            Lidfeat=self.get_id_feature_dist_loss(all_features, all_target, confidenced_id_mask)
            if Lidfeat.item()<0:  
                self.logger.info("Lidfeat : {}".format(Lidfeat.item()))
            
            self.losses_pap_id.update(Lidfeat.item(), confidenced_id_mask.sum())     
                
            Loodfeat=0.
            if ood_mask.sum()>0:        
                Loodfeat=self.get_ood_feature_dist_loss(ul_feature_weak,ul_feature_strong,ood_mask) 
                if Loodfeat.item()<0:  
                    self.logger.info("Loodfeat : {}".format(Loodfeat.item()))
                    
                self.losses_pap_ood.update(Loodfeat.item(), ood_mask.sum())   
            pap_loss= self.pap_loss_weight*(Lidfeat+Loodfeat)
            # loss_dict.update({"pap_loss": pap_loss}) 
            
        loss=cons_loss+cls_loss+pap_loss
        # loss = sum(loss_dict.values())
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(cls_loss.item(), inputs_x.size(0))
        self.losses_u.update(cons_loss.item(), inputs_u.size(0)) 
        
        # self.losses_pap.update(pap_loss.item(),inputs_x.size(0)+confidenced_id_mask.sum()+ood_mask.sum())
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_u.val))
            self.logger.info('========== Loss_pap:{:>5.4f} Loss_pap_id:{:>5.4f} Loss_pap_ood:{:>5.4f} ==========='.format(self.losses_pap.val,self.losses_pap_id.val,self.losses_pap_ood.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()  
   
    def get_id_feature_dist_loss(self,features, targets,id_mask):         
        prototypes = self.queue.prototypes  # (K, D)
        # pair_dist=-1  *torch.cdist(features,self.dl_center)  
        pair_dist=-1  *torch.cdist(features,prototypes)  
        logits=torch.div(pair_dist, self.feature_loss_temperature)  
        mask_same_c=torch.eq(\
            targets.contiguous().view(-1, 1).cuda(), \
            torch.tensor([i for i in range(self.num_classes)]).contiguous().view(-1, 1).cuda().T).float()
        id_mask=id_mask.expand(mask_same_c.size(1),-1).T # torch.Size([10,192]) # old 
        mask_same_c*=id_mask
        log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
        log_prob_pos = log_prob * mask_same_c # mask掉其他的负类
        loss = - log_prob_pos.sum() / mask_same_c.sum()   
        if loss.item()<0:
            print("id loss < 0")
        return loss       
    
    def get_ood_feature_dist_loss(self,features_u,features_u2,ood_mask):         
        features=torch.cat([features_u,features_u2],0)        
        prototypes = self.queue.prototypes  # (K, D)
        all_features=torch.cat([features,prototypes],0) # [138,64]
        # all_features=torch.cat([features,self.dl_center],0) # [138,64]
        mask_same_c=torch.cat([torch.eye(features_u.size(0)),torch.eye(features_u.size(0)),torch.zeros((features_u.size(0),self.num_classes))],dim=1)
        mask_same_c=torch.cat([mask_same_c,mask_same_c],dim=0).cuda()    # [128,138]       
        pair_dist=-1 *torch.cdist(features,all_features)   # [128,138]  
        logits=torch.div(pair_dist, self.feature_loss_temperature)    # [128,138] 
        ood_mask=torch.cat([ood_mask,ood_mask],0)
        ood_mask=ood_mask.expand(mask_same_c.size(1),-1).T
        mask_same_c*=ood_mask
        log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
        log_prob_pos = log_prob * mask_same_c 
        loss = - log_prob_pos.sum() / mask_same_c.sum()   
        if loss.item()<0:
            print("ood loss < 0")
        return loss   
    
    def mix_up(self,l_images,ul_images):                 
        with torch.no_grad():     
            len_l=l_images.size(0)
            len_aux=ul_images.size(0)
            if len_aux==0: 
                return l_images
            elif len_aux>len_l:
                ul_images=ul_images[:len_l]
            elif len_aux<len_l:
                extend_num= math.ceil(len_l/len_aux)
                tmp=[ul_images]*extend_num
                ul_images=torch.cat(tmp,dim=0)[:len_l]

            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1.0 - lam)
            rand_idx = torch.randperm(l_images.size(0)) 
            mixed_images = lam * l_images + (1.0 - lam) * ul_images[rand_idx]
            return mixed_images  
        
    def operate_after_epoch(self): 
        if self.iter<=self.warmup_iter:            
            self.detect_ood()
        if self.iter==self.warmup_iter:
            self.save_checkpoint(file_name="warmup_model.pth")
            # self._rebuild_models()
            # self._rebuild_optimizer(self.model)
        ood_pre,ood_rec=self.ood_detect_fusion.get_pre_per_class()[1],self.ood_detect_fusion.get_rec_per_class()[1]
        id_pre,id_rec=self.id_detect_fusion.get_pre_per_class()[1],self.id_detect_fusion.get_rec_per_class()[1]
        self.logger.info("== ood_prec:{:>5.3f} ood_rec:{:>5.3f} id_prec:{:>5.3f} id_rec:{:>5.3f}".\
            format(ood_pre*100,ood_rec*100,id_pre*100,id_rec*100))
        self.logger.info('=='*40)    
        
        if self.iter<self.warmup_iter:   
            self.ood_detect_fusion.reset() 
            self.id_detect_fusion.reset()
            
    def _rebuild_models(self):
        model = self.build_model(self.cfg) 
        self.model = model.cuda()
        self.ema_model = EMAModel(
            self.model,
            self.cfg.MODEL.EMA_DECAY,
            self.cfg.MODEL.EMA_WEIGHT_DECAY,
        )
        # .cuda()  
        # self.projector=Projector(self.cfg).cuda()        

    def prepare_feat(self,dataloader,return_confidence=False):
        model=self.get_val_model().eval()
        n=dataloader.dataset.total_num
        feat=torch.zeros((n,self.feature_dim)) 
        targets_y=torch.zeros(n).long()
        confidence=torch.zeros(n)  
        probs=torch.zeros(n,self.num_classes) 
        with torch.no_grad():
            for batch_idx,(inputs, targets, idx) in enumerate(dataloader):
                if isinstance(inputs, list):
                    inputs=inputs[0]
                inputs, targets = inputs.cuda(), targets.cuda()
                
                # encoding=self.model(inputs,return_encoding=True)
                # outputs=self.model(encoding,return_projected_feature=True) 
                
                outputs=self.model(inputs,return_encoding=True)
                # outputs=self.projector(outputs)
                logits=self.model(outputs,classifier=True)
                prob = torch.softmax(logits.detach(), dim=-1) 
                max_probs, pred_class = torch.max(prob, dim=-1)  
                
                feat[idx] =   outputs.cpu()  
                targets_y[idx] = targets.cpu()                 
                confidence[idx]=max_probs.cpu()
                probs[idx]=prob.cpu()
                
        # feat=torch.cat(feat,dim=0)
        # targets_y=torch.cat(targets_y,dim=0)
        if return_confidence:
            return feat,targets_y,[confidence,probs]
            
        return feat,targets_y
    
    
    def detect_ood(self,):
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)        
        # 2023.11.7在分类器前一层的特征norm后效果更好
        l_feat,l_y=self.prepare_feat(self.test_labeled_trainloader)
        l_feat=normalizer(l_feat)
        u_feat,u_y=self.prepare_feat(self.test_unlabeled_trainloader)
        u_feat=normalizer(u_feat)
        # du_gt=torch.zeros(self.ul_num) 
        id_mask=torch.zeros(self.ul_num).long().cuda()
        ood_mask=torch.zeros(self.ul_num).long().cuda()
        du_gt=(u_y>=0).float().long().cpu().numpy()
        index = faiss.IndexFlatL2(l_feat.shape[1])
        index.add(l_feat)
        D, _ = index.search(u_feat, self.k) 
        novel = -D[:,-1] # -最大的距离
        D2, _ = index.search(l_feat, self.k)
        known= - D2[:,-1] # -最大的距离 
        known.sort() # 从小到大排序 负号
        thresh = known[50] #known[50] #  known[round(0.05 * self.l_num)]        
        id_masks= (torch.tensor(novel)>=thresh).long()
        ood_masks=1-id_masks 
        self.id_masks= id_masks
        self.ood_masks=1-id_masks
        self.id_masks=self.id_masks.cuda()
        self.ood_masks=self.ood_masks.cuda()
        self.id_detect_fusion.reset()
        self.ood_detect_fusion.reset()
        self.id_detect_fusion.update(id_masks.numpy(),du_gt) 
        self.ood_detect_fusion.update(ood_masks.numpy(),1-du_gt)  
        # all_features=torch.zeros(self.l_num+self.ul_num
        #                          ,self.feature_dim).cuda()
        # du_features=torch.zeros(self.ul_num,
        #                         self.feature_dim).cuda() 
        # all_domain_y=torch.cat([torch.zeros(self.ul_num),
        #                         torch.ones(self.l_num)],dim=0).long().cuda()
        # du_gt=torch.zeros(self.ul_num).long().cuda()
        
        # id_mask=torch.zeros(self.ul_num).long().cuda()
        # ood_mask=torch.zeros(self.ul_num).long().cuda()
        # with torch.no_grad():
        #     for  i, ((inputs,_), target, idx) in enumerate(self.unlabeled_trainloader):
        #         inputs=inputs.cuda() 
        #         target=target.cuda()
        #         feat=self.model(inputs,return_encoding=True)
        #         feat=self.projector(feat)
        #         du_features[idx]=feat.detach()
        #         all_features[idx]=feat.detach()                 
        #         ones=torch.ones_like(target).long().cuda()
        #         zeros=torch.zeros_like(target).long().cuda()
        #         gt=torch.where(target>=0,ones,zeros)
        #         du_gt[idx]=gt
        #     for  i, (inputs, _, idx) in enumerate(self.labeled_trainloader):
        #         inputs=inputs.cuda()                 
        #         feat=self.model(inputs,return_encoding=True)
        #         feat=self.projector(feat)               
        #         all_features[idx+self.ul_num]=feat.detach()   
        # for i in range(self.update_domain_y_iter): # 迭代10次更新
        #     select_index=torch.nonzero(id_mask == 0, as_tuple=False).squeeze(1)
        #     if select_index.size(0)==0:
        #         break
        #     ood_feat=du_features[select_index]
        #     # [B, K]        
        #     sim_matrix = torch.mm(ood_feat, all_features.t())
        #     sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1) # 
        #     # new_d_y=[]
        #     # for item in sim_indices: # [n,50] 
        #     d_y=all_domain_y[sim_indices]        
        #     count_idnn=torch.count_nonzero(d_y,dim=1)        
        #     ones=torch.ones_like(count_idnn).cuda()
        #     zeros=torch.zeros_like(count_idnn).cuda()
        #     new_d_y_id = torch.where(count_idnn >= self.k//2,ones,zeros).long().cuda() 
        #     new_d_y_ood = torch.where(count_idnn <= self.k//3,ones,zeros).long().cuda() 
        #     all_domain_y[select_index]=new_d_y_id 
        #     id_mask[select_index]=new_d_y_id
        #     ood_mask[select_index]=new_d_y_ood
        #     ood_mask[select_index]*=(1-new_d_y_id)  # 避免重复
        # # self.domain_y=du_domain_y   
        # self.id_masks= id_mask
        # self.ood_masks=ood_mask
        # self.ood_detect_fusion.update(ood_mask,1-du_gt)    
        # self.id_detect_fusion.update(id_mask,du_gt)        
        return  
    
    def save_checkpoint(self,file_name=""):
        if file_name=="":
            file_name="checkpoint.pth" if self.iter!=self.max_iter else "model_final.pth"
        torch.save({
                    'model': self.model.state_dict(),
                    'ema_model': self.ema_model.state_dict(),
                    'iter': self.iter, 
                    'best_val': self.best_val, 
                    'best_val_iter':self.best_val_iter, 
                    'best_val_test': self.best_val_test,
                    'optimizer': self.optimizer.state_dict(),
                    # 'projector': self.projector.state_dict(),
                    "prototypes":self.queue.prototypes,
                    # 'bank':self.queue.bank,
                    'id_masks':self.id_masks,
                    'ood_masks':self.ood_masks,                   
                },  os.path.join(self.model_dir, file_name))
        return    
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")

        state_dict = torch.load(resume)
        
        if 'warmup_model.pth' in resume:
            # 加载warmup
            pass
        else:
            
            # load model
            self.model.load_state_dict(state_dict["model"])

            # load ema model 
            self.ema_model.load_state_dict(state_dict["ema_model"])

            # load optimizer and scheduler
            self.optimizer.load_state_dict(state_dict["optimizer"]) 
            # self.projector.load_state_dict(state_dict["projector"])
        try:
            self.id_masks=state_dict['id_masks']
            self.ood_masks=state_dict['ood_masks'] 
        except:
            self.logger.warning("the id_masks and ood_masks of resume file are none!")
        
        self.queue.prototypes=state_dict["prototypes"]
        # self.queue.bank=state_dict['bank']
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.val_iter) 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        )
    def knn_ood_detect(self,ul_weak):          
        with torch.no_grad():
            feat=self.all_features 
            sim_matrix = torch.mm(ul_weak, feat.t())
            sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1) #  
            d_y=self.all_domain_y[sim_indices]   
            count_idnn=torch.count_nonzero(d_y,dim=1)        
            ones=torch.ones_like(count_idnn).cuda()
            zeros=torch.zeros_like(count_idnn).cuda()
            id_mask = torch.where(count_idnn >= self.k//2,ones,zeros).long().cuda() 
            ood_mask = torch.where(count_idnn <= (self.k)//3,ones,zeros).long().cuda() 
            # ood_mask=1-id_mask            
            return id_mask,ood_mask 
    
    def get_dl_center(self,): 
        features=[]
        labels=[]
        with torch.no_grad():
            for  i, ((inputs,_), targets, _) in enumerate(self.dl_center_loader):
                inputs=inputs.cuda()
                targets=targets.long().cuda()
                feat=self.model(inputs,return_encoding=True)
                # feat=self.projector(feat)
                feat=self.model.project_feature(feat)
                features.append(feat.detach())                
                self.queue.enqueue(feat.clone().detach(), targets.clone().detach())
                labels.append(targets)
            features=torch.cat(features,dim=0)
            labels=torch.cat(labels,dim=0)
            uniq_c = torch.unique(labels)
            for c in uniq_c:
                c = int(c)
                select_index = torch.nonzero(
                    labels == c, as_tuple=False).squeeze(1)
                embedding_temp = features[select_index]  
                mean = embedding_temp.mean(dim=0) 
                self.dl_center[c] = mean
        return 
     
    def update_center(self,features,labels):
        with torch.no_grad():
            if len(labels) > 0:
                uniq_c = torch.unique(labels)
                for c in uniq_c:
                    c = int(c)
                    select_index = torch.nonzero(
                        labels == c, as_tuple=False).squeeze(1)
                    embedding_temp = features[select_index]  
                    mean = embedding_temp.mean(dim=0)
                    var = embedding_temp.var(dim=0, unbiased=False)
                    n = embedding_temp.numel() / embedding_temp.size(1)
                    if n > 1:
                        var = var * n / (n - 1)
                    else:
                        var = var 
                if torch.count_nonzero(self.dl_center[c])>0:
                    self.dl_center[c] =(1 - self.center_decay_ratio)* mean    +  \
                        self.center_decay_ratio* self.dl_center[c]
                else:
                    self.dl_center[c] = mean
        return