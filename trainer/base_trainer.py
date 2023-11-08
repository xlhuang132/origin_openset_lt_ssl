   
import logging
from operator import mod
from tkinter import W
import torch 
import numpy as np
from dataset.build_dataloader import build_dataloader
from loss.build_loss import build_loss 
import models 
import time 
import torch.optim as optim
from models.feature_queue import FeatureQueue
import os   
import datetime
import torch.nn.functional as F
from utils import AverageMeter, accuracy, create_logger,\
    plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch,plot_acc_over_epoch
from utils.build_optimizer import get_optimizer, get_scheduler
 
from utils import FusionMatrix
from utils.ema_model import EMAModel

class BaseTrainer():
    def __init__(self,cfg):
        
        self.cfg=cfg
        self.logger, _ = create_logger(cfg)        
        self.path,self.model_dir,self.pic_dir =prepare_output_path(cfg,self.logger)
        self.num_classes=cfg.DATASET.NUM_CLASSES
        self.batch_size=cfg.DATASET.BATCH_SIZE
        # =================== build model =============
        self.ema_enable=False
        self.model = models.__dict__[cfg.MODEL.NAME](cfg)
        self.ema_model = EMAModel(
            self.model,
            cfg.MODEL.EMA_DECAY,
            cfg.MODEL.EMA_WEIGHT_DECAY, 
        )
        self.model=self.model.cuda()
        # self.ema_model=self.ema_model.cuda()
        # =================== build dataloader =============
        self.build_data_loaders()
        # =================== build criterion ==============
        self.build_loss()
        # ========== build optimizer ===========         
        self.optimizer = get_optimizer(cfg, self.model)
        
        # ========== build dataloader ==========     
        
        self.max_epoch=cfg.MAX_EPOCH
        self.func = torch.nn.Softmax(dim=1) 
        
        # ========== accuracy history =========
        self.test_accs=[]
        self.val_accs=[]
        self.train_accs=[]
        self.test_group_accs=[]
        self.val_group_accs=[]
        self.train_group_accs=[]
        
        # ========== loss history =========
        self.train_losses=[]
        self.val_losses=[]
        self.test_losses=[]
        
        self.conf_thres=cfg.ALGORITHM.CONFIDENCE_THRESHOLD   
        
        self.iter=0
        self.best_val=0
        self.best_val_iter=0
        self.best_val_test=0
        self.start_iter=1
        self.max_iter=cfg.MAX_ITERATION+1
        self.val_iter=cfg.VAL_ITERATION
        self.pretraining=False  
        self.warmup_iter=0
        self.save_epoch=cfg.SAVE_EPOCH
        self.warmup_enable=cfg.ALGORITHM.PRE_TRAIN.ENABLE
        
    @classmethod
    def build_model(cls, cfg)  :
        model = models.__dict__[cfg.MODEL.NAME](cfg)
        return model
    
    @classmethod
    def build_optimizer(cls, cfg , model )  :
        return get_optimizer(cfg, model)
    
    def build_loss(self):
        self.l_criterion,self.ul_criterion,self.val_criterion = build_loss(self.cfg)
        return 
    
    def build_data_loaders(self,)  :
        # l_loader, ul_loader, val_loader, test_loader           
        # self.labeled_trainloader, self.unlabeled_trainloader, self.val_loader, self.test_loader = build_data_loaders(cfg) 
        # self.unlabeled_train_iter = iter(self.unlabeled_trainloader)        
        # self.labeled_train_iter = iter(self.labeled_trainloader)   
        dataloaders=build_dataloader(self.cfg,self.logger)
        self.domain_trainloader=dataloaders[0]
        self.labeled_trainloader=dataloaders[1]
        self.labeled_train_iter=iter(self.labeled_trainloader)        
        # DU               
        self.unlabeled_trainloader=dataloaders[2]
        self.unlabeled_train_iter=iter(self.unlabeled_trainloader)   
        self.val_loader=dataloaders[3]
        self.test_loader=dataloaders[4]
        if len(dataloaders)>5 and dataloaders[5] is not None:            
            self.dual_labeled_trainloader=dataloaders[5]
            self.dual_l_iter=iter(self.dual_labeled_trainloader)
        if len(dataloaders)>6:            
            self.pre_train_loader=dataloaders[6]
            self.pre_train_iter=iter(self.pre_train_loader)
        return  
    
    def train_step(self,pretraining=False):
        pass
    
    def train(self,):
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()      
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.epoch= (self.iter // self.val_iter)+1  
        start_time = time.time()   
        for self.iter in range(self.start_iter, self.max_iter):
            self.pretraining= self.warmup_enable and self.iter<=self.warmup_iter 
            return_data=self.train_step(self.pretraining)
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
                start_time = time.time()    
                self.operate_after_epoch()
                
                
        self.plot()       
        return
    
    def operate_after_epoch(self):
        self.logger.info('=='*30)
        pass
    def get_val_model(self,):
        return self.model
    
    def _rebuild_models(self):
        model = self.build_model(self.cfg) 
        self.model = model.cuda()
        self.ema_model = EMAModel(
            self.model,
            self.cfg.MODEL.EMA_DECAY,
            self.cfg.MODEL.EMA_WEIGHT_DECAY,
        )
        # .cuda() 
        

    def _rebuild_optimizer(self, model):
        self.optimizer = self.build_optimizer(self.cfg, model)
       
    def evaluate(self,return_group_acc=False,return_class_acc=False):  
        eval_model=self.get_val_model() 
        val_loss, val_acc,val_group_acc,val_class_acc = self.eval_loop(eval_model,self.val_loader, self.val_criterion)
        test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_loop(eval_model,self.test_loader, self.val_criterion)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.val_group_accs.append(val_group_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)
        self.test_group_accs.append(test_group_acc)
        
        if return_group_acc:
            if return_class_acc:
                return val_acc,test_acc,test_group_acc,test_class_acc
            else:
                return val_acc,test_acc,test_group_acc
        if return_class_acc:
            return val_acc,test_acc,test_class_acc
        return [val_acc,test_acc]
    
    def get_test_data_pred_gt_feat(self):
        model=self.get_val_model()
        model.eval()
        pred=[]
        gt=[]
        feat=[]
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)
                feature=model(inputs,return_encoding=True)
                score_result = self.func(outputs)
                now_result = torch.argmax(score_result, 1)   
                gt.append(targets.cpu())   
                pred.append(now_result.cpu())
                feat.append(feature.cpu())
            pred=torch.cat(pred,dim=0)
            gt=torch.cat(gt,dim=0)
            feat=torch.cat(feat,dim=0)
        return gt,pred,feat
    
    def get_train_dl_data_pred_gt_feat(self):
        model=self.get_val_model()
        model.eval()
        pred=[]
        gt=[]
        feat=[]
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(self.labeled_trainloader):
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)
                feature=model(inputs,return_encoding=True)
                score_result = self.func(outputs)
                now_result = torch.argmax(score_result, 1)   
                gt.append(targets.cpu())   
                pred.append(now_result.cpu())
                feat.append(feature.cpu())
            pred=torch.cat(pred,dim=0)
            gt=torch.cat(gt,dim=0)
            feat=torch.cat(feat,dim=0)
        return gt,pred,feat
                
    
    def eval_loop(self,model,valloader,criterion):
        losses = AverageMeter() 
        # switch to evaluate mode
        model.eval()
 
        fusion_matrix = FusionMatrix(self.num_classes)
        func = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(valloader):
                # measure data loading time 

                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)
                if len(outputs)==2:
                    outputs=outputs[0]
                loss = criterion(outputs, targets)

                # measure accuracy and record loss 
                losses.update(loss.item(), inputs.size(0)) 
                score_result = func(outputs)
                now_result = torch.argmax(score_result, 1) 
                fusion_matrix.update(now_result.cpu().numpy(), targets.cpu().numpy())
                 
        group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS)
        class_acc=fusion_matrix.get_acc_per_class()
        acc=fusion_matrix.get_accuracy()    
        return (losses.avg, acc, group_acc,class_acc)
  
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
                    'optimizer': self.optimizer.state_dict()
                },  os.path.join(self.model_dir, file_name))
        return 
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")

        state_dict = torch.load(resume)
        # load model
        self.model.load_state_dict(state_dict["model"])

        # load ema model 
        self.ema_model.load_state_dict(state_dict["ema_model"])

        # load optimizer and scheduler
        self.optimizer.load_state_dict(state_dict["optimizer"])   
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.val_iter) 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        )
  
    def extract_feature(self):
        model=self.get_val_model()
        model.eval()
        with torch.no_grad():
            labeled_feat=[] 
            labeled_y=[] 
            labeled_idx=[]
            for i,data in enumerate(self.labeled_trainloader):
                inputs_x, targets_x,idx=data[0],data[1],data[2]
                inputs_x=inputs_x.cuda()
                feat=model(inputs_x,return_encoding=True)
                labeled_feat.append(feat.cpu())
                labeled_y.append(targets_x)
                labeled_idx.append(idx)
            labeled_feat=torch.cat(labeled_feat,dim=0).cpu()
            labeled_y=torch.cat(labeled_y,dim=0) 
            labeled_idx=torch.cat(labeled_idx,dim=0) 
            unlabeled_feat=[]
            unlabeled_y=[] 
            unlabeled_idx=[]
            for i,data in enumerate(self.unlabeled_trainloader):
                (inputs_u,_)=data[0]
                inputs_u=inputs_u.cuda()
                target=data[1]      
                idx=data[2]      
                feat=model(inputs_u,return_encoding=True)
                unlabeled_feat.append(feat.cpu())
                unlabeled_y.append(target)
                unlabeled_idx.append(idx)
            unlabeled_feat=torch.cat(unlabeled_feat,dim=0)
            unlabeled_y=torch.cat(unlabeled_y,dim=0) 
            unlabeled_idx=torch.cat(unlabeled_idx,dim=0) 
            test_feat=[]
            test_y=[]
            test_idx=[]
            for i,data in enumerate(self.test_loader):
                inputs_x, target,idx=data[0],data[1],data[2]
                inputs_x=inputs_x.cuda()
                feat=model(inputs_x,return_encoding=True)
                test_feat.append(feat.cpu())            
                test_y.append(target)        
                test_idx.append(idx)
            test_feat=torch.cat(test_feat,dim=0)
            test_y=torch.cat(test_y,dim=0) 
            return (labeled_feat,labeled_y,labeled_idx),(unlabeled_feat,unlabeled_y,unlabeled_idx),(test_feat,test_y,test_idx)
    
    def compute_mean_cov(self,feat,y): # 计算均值+协方差
        uniq_c = torch.unique(y)
        means=torch.zeros(self.num_classes,feat.size(1))
        covs=torch.zeros(self.num_classes,feat.size(1))
        for c in uniq_c:
            c = int(c)
            if c==-1:continue
            select_index = torch.nonzero(
                y == c, as_tuple=False).squeeze(1)
            embedding_temp = embedding[select_index]  
            mean = embedding_temp.mean(dim=0)
            var = embedding_temp.var(dim=0, unbiased=False)
            means[c]=mean
            covs[c]=var
        return means,covs
        
    def plot(self):
        plot_group_acc_over_epoch(group_acc=self.train_group_accs,title="Train Group Average Accuracy",save_path=os.path.join(self.pic_dir,'train_group_acc.jpg'))
        plot_group_acc_over_epoch(group_acc=self.val_group_accs,title="Val Group Average Accuracy",save_path=os.path.join(self.pic_dir,'val_group_acc.jpg'))
        plot_group_acc_over_epoch(group_acc=self.test_group_accs,title="Test Group Average Accuracy",save_path=os.path.join(self.pic_dir,'test_group_acc.jpg'))
        plot_acc_over_epoch(self.train_accs,title="Train average accuracy",save_path=os.path.join(self.pic_dir,'train_acc.jpg'),)
        plot_acc_over_epoch(self.test_accs,title="Test average accuracy",save_path=os.path.join(self.pic_dir,'test_acc.jpg'),)
        plot_acc_over_epoch(self.val_accs,title="Val average accuracy",save_path=os.path.join(self.pic_dir,'val_acc.jpg'),)
        plot_loss_over_epoch(self.train_losses,title="Train Average Loss",save_path=os.path.join(self.pic_dir,'train_loss.jpg'))
        plot_loss_over_epoch(self.val_losses,title="Val Average Loss",save_path=os.path.join(self.pic_dir,'val_loss.jpg'))
        plot_loss_over_epoch(self.test_losses,title="Test Average Loss",save_path=os.path.join(self.pic_dir,'test_loss.jpg'))
    
    def get_class_counts(self,dataset):
        """
            Sort the class counts by class index in an increasing order
            i.e., List[(2, 60), (0, 30), (1, 10)] -> np.array([30, 10, 60])
        """
        return np.array(dataset.num_per_cls_list)
        # class_count = dataset.num_samples_per_class

        # # sort with class indices in increasing order
        # class_count.sort(key=lambda x: x[0])
        # per_class_samples = np.asarray([float(v[1]) for v in class_count])
        # return per_class_samples
    
    def get_label_dist(self, dataset=None, normalize=None):
        """
            normalize: ["sum", "max"]
        """
        if dataset is None:
            dataset = self.labeled_trainloader.dataset

        class_counts = torch.from_numpy(self.get_class_counts(dataset)).float()
        class_counts = class_counts.cuda()

        if normalize:
            assert normalize in ["sum", "max"]
            if normalize == "sum":
                return class_counts / class_counts.sum()
            if normalize == "max":
                return class_counts / class_counts.max()
        return class_counts

    def build_labeled_loss(self, cfg , warmed_up=False)  :
        loss_type = cfg.MODEL.LOSS.LABELED_LOSS
        num_classes = cfg.MODEL.NUM_CLASSES
        assert loss_type == "CrossEntropyLoss"

        class_count = self.get_label_dist(device=self.device)
        per_class_weights = None
        if cfg.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE and warmed_up:
            loss_override = cfg.MODEL.LOSS.COST_SENSITIVE.LOSS_OVERRIDE
            beta = cfg.MODEL.LOSS.COST_SENSITIVE.BETA
            if beta < 1:
                # effective number of samples;
                effective_num = 1.0 - torch.pow(beta, class_count)
                per_class_weights = (1.0 - beta) / effective_num
            else:
                per_class_weights = 1.0 / class_count

            # sum to num_classes
            per_class_weights = per_class_weights / torch.sum(per_class_weights) * num_classes

            if loss_override == "":
                # CE loss
                loss_fn = build_loss(
                    cfg, loss_type, class_count=class_count, class_weight=per_class_weights
                )

            elif loss_override == "LDAM":
                # LDAM loss
                loss_fn = build_loss(
                    cfg, "LDAMLoss", class_count=class_count, class_weight=per_class_weights
                )

            else:
                raise ValueError()
        else:
            loss_fn = build_loss(
                cfg, loss_type, class_count=class_count, class_weight=per_class_weights
            )

        return loss_fn
