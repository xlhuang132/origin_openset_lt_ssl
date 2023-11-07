import numpy as np
import torch
import os
import copy
def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1 # 所有的数据按batch 分成了多片，最后一片不是batch的整数倍
    offsets = interleave_offsets(batch, nu) # 得到差分的interleave
    # ==== hxl code =====
    # total_len=sum([len(item) for item in xy])
    # if total_len<nu *batch+batch:
    #     xy[-1]=torch.cat((xy[-1],torch.zeros(nu*batch+batch-total_len,xy[0].size(1),xy[0].size(2),xy[0].size(3)).cuda()),dim=0)
    # =================== p:[0,nu] offsets:[0 10 20 31 42 63 64]
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy] # 对xy里面的每片 取出每个区间的值
    for i in range(1, nu + 1): # [1,nu]
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i] #交换这两个点的位置  
    return [torch.cat(v, dim=0) for v in xy] # 再把每个区间拼接起来

def load_checkpoint(model,model_path,cfg=None):
    assert os.path.exists(model_path)        
    print(f"===> Loading checkpoint '{model_path}'")
    checkpoint = torch.load(model_path)               
    model.load_state_dict(checkpoint["model"])    
    return model

def get_group_splits(cfg): # 返回 many medium few 的 class id 列表
    num_classes=cfg.DATASET.NUM_CLASSES
    group_splits=cfg.DATASET.GROUP_SPLITS
    assert num_classes==sum(group_splits)
    id_splits=[]
    l=0
    class_ids=np.array([i for i in range(num_classes)])
    for item in group_splits:
        id_splits.append(class_ids[l:l+item])
        l+=item
    return id_splits
 
def get_DL_dataset_path(cfg,dataset=None,):
    dataset = cfg.DATASET.NAME if not dataset else dataset
    path=os.path.join(cfg.OUTPUT_DIR, dataset)
    return path

def get_DL_dataset_alg_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None):
    parent_path=get_DL_dataset_path(cfg,dataset=dataset)
    algorithm_name=cfg.ALGORITHM.NAME   if not algorithm else algorithm 
    model_name=cfg.MODEL.NAME
    labeled_loss_type=cfg.MODEL.LOSS.LABELED_LOSS_CLASS_WEIGHT_TYPE if not labeled_loss_type else labeled_loss_type
    if labeled_loss_type and labeled_loss_type!='None':
        algorithm_name=algorithm_name+labeled_loss_type
    path=os.path.join(parent_path, algorithm_name, model_name)
    return path

def get_DL_dataset_alg_DU_dataset_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None):
    parent_path=get_DL_dataset_alg_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type)
    num_labeled_head=cfg.DATASET.DL.NUM_LABELED_HEAD if not num_labeled_head else num_labeled_head
    imb_factor_l=cfg.DATASET.DL.IMB_FACTOR_L if not imb_factor_l else imb_factor_l
    num_unlabeled_head=cfg.DATASET.DU.ID.NUM_UNLABELED_HEAD if not num_unlabeled_head else num_unlabeled_head
    imb_factor_ul=cfg.DATASET.DU.ID.IMB_FACTOR_UL  if not imb_factor_ul else imb_factor_ul
    # DL + DU_ID数据设置
    DL_DU_ID_setting='DL-{}-IF-{}-DU{}-IF_U-{}'.format(num_labeled_head,imb_factor_l,num_unlabeled_head,imb_factor_ul)
    path=os.path.join(parent_path, DL_DU_ID_setting)
    return path

def get_DL_dataset_alg_DU_dataset_OOD_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None,
             ood_dataset=None,ood_r=None
             ): 
    parent_path=get_DL_dataset_alg_DU_dataset_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type,
             num_labeled_head=num_labeled_head,imb_factor_l=imb_factor_l,num_unlabeled_head=num_unlabeled_head,imb_factor_ul=imb_factor_ul)
    ood_dataset=cfg.DATASET.DU.OOD.DATASET if not ood_dataset else ood_dataset
    ood_r=cfg.DATASET.DU.OOD.RATIO if not ood_r else ood_r
    # DL + DU_ID数据设置
    OOD_setting='OOD-{}-r-{:.2f}'.format(ood_dataset,ood_r)
    path=os.path.join(parent_path, OOD_setting)
    return path

def get_warmup_model_path(cfg,warmup_model_root="warmup_model"):
    file_path=get_root_path(cfg) 
    idx=file_path.find("/")
    file_path=warmup_model_root+file_path[idx:]
    warmup_epoch=cfg.ALGORITHM.PRE_TRAIN.WARMUP_EPOCH
    warmup_model_path=os.path.join(file_path,"models","epoch_{}.pth".format(warmup_epoch))
    return warmup_model_path

def get_ablation_file_name(cfg):
    file_name="" 
    if cfg.ALGORITHM.ABLATION.DUAL_BRANCH:
        file_name+="DUAL_BRANCH"    
    if cfg.ALGORITHM.ABLATION.MIXUP:
        file_name+="-MIXUP" 
    if cfg.ALGORITHM.ABLATION.OOD_DETECTION:
        file_name+="-OOD_DETECT"
    if cfg.ALGORITHM.ABLATION.PAP_LOSS:
        file_name+="-PAP_LOSS" 
    if file_name=="":
        file_name='A_Naive'
    return file_name
def get_root_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None,
             ood_dataset=None,ood_r=None,
             sampler=None,sampler_mixup=None,dual_sampler_enable=None,dual_sampler=None,dual_sampler_mixup=None,
             Branch_setting=None, # 优先
             ):
    parent_path=get_DL_dataset_alg_DU_dataset_OOD_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type,
             num_labeled_head=num_labeled_head,imb_factor_l=imb_factor_l,
             num_unlabeled_head=num_unlabeled_head,imb_factor_ul=imb_factor_ul,
             ood_dataset=ood_dataset,ood_r=ood_r
             )
    # DL双单通道设置 + 单双是否mixup设置 
    if not Branch_setting:
        sampler=cfg.DATASET.SAMPLER.NAME if not sampler else sampler
        sampler_mixup=cfg.ALGORITHM.BRANCH1_MIXUP if sampler_mixup==None else sampler_mixup
        dual_sampler_enable=cfg.DATASET.DUAL_SAMPLER.ENABLE if dual_sampler_enable==None else dual_sampler_enable
        dual_sampler=cfg.DATASET.DUAL_SAMPLER.NAME if not dual_sampler else dual_sampler
        dual_sampler_mixup=cfg.ALGORITHM.BRANCH2_MIXUP if dual_sampler_mixup==None else dual_sampler_mixup
        if sampler_mixup:
            Branch_setting='{}_mixup'.format(sampler) 
        else:
            Branch_setting='{}'.format(sampler) 
        if dual_sampler_enable:
            if dual_sampler_mixup:
                Branch_setting+='-{}_mixup'.format(dual_sampler) 
            else:
                Branch_setting+='-{}'.format(dual_sampler)  
        
    # path=os.path.join(parent_path,Branch_setting)
    path=parent_path
    if cfg.ALGORITHM.ABLATION.ENABLE:
        file_name=get_ablation_file_name(cfg)
        path=os.path.join(path,file_name)
    return path 
def prepare_output_path(cfg,logger):
    # 准备models和pic输出文件
    path= get_root_path(cfg)
    model_dir = os.path.join(path ,"models") 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        ) 
    print("=> output model will be saved in {}".format(model_dir)) 
    pic_dir= os.path.join(path ,"pic") 
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir) 
    return path,model_dir,pic_dir 


def create_ema_model(model):
    ema_model=copy.deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return ema_model



def compute_mean_std_cha(train_gt,train_pred,train_feat,test_gt,test_pred,test_feat):
    # 算所有的 
    uniq_c=torch.unique(train_gt) 
    num_classes=len(uniq_c)
    feature_dim=train_feat.size(1)
    train_gt_mean=torch.zeros(num_classes,feature_dim)
    train_pred_mean=torch.zeros(num_classes,feature_dim)
    test_gt_mean=torch.zeros(num_classes,feature_dim)
    test_pred_mean=torch.zeros(num_classes,feature_dim)
    train_gt_var=torch.zeros(num_classes,feature_dim)
    train_pred_var=torch.zeros(num_classes,feature_dim)
    test_gt_var=torch.zeros(num_classes,feature_dim)
    test_pred_var=torch.zeros(num_classes,feature_dim)
    for c in uniq_c:
        c = int(c)
        select_index = torch.nonzero(
            train_gt == c, as_tuple=False).squeeze(1)
        embedding_temp = train_feat[select_index]  
        mean = embedding_temp.mean(dim=0)
        var = embedding_temp.var(dim=0, unbiased=False)
        n = embedding_temp.numel() / embedding_temp.size(1)
        if n > 1:
            var = var * n / (n - 1)
        else:
            var = var 
        train_gt_mean[c]=mean
        train_gt_var[c]=var
        
        select_index = torch.nonzero(
            train_pred == c, as_tuple=False).squeeze(1)
        embedding_temp = train_feat[select_index]  
        mean = embedding_temp.mean(dim=0)
        var = embedding_temp.var(dim=0, unbiased=False)
        n = embedding_temp.numel() / embedding_temp.size(1)
        if n > 1:
            var = var * n / (n - 1)
        else:
            var = var 
        train_pred_mean[c]=mean
        train_pred_var[c]=var
        
        select_index = torch.nonzero(
            test_gt == c, as_tuple=False).squeeze(1)
        embedding_temp = test_feat[select_index]  
        mean = embedding_temp.mean(dim=0)
        var = embedding_temp.var(dim=0, unbiased=False)
        n = embedding_temp.numel() / embedding_temp.size(1)
        if n > 1:
            var = var * n / (n - 1)
        else:
            var = var 
        test_gt_mean[c]=mean
        test_gt_var[c]=var
        
        select_index = torch.nonzero(
            test_pred == c, as_tuple=False).squeeze(1)
        embedding_temp = test_feat[select_index]  
        mean = embedding_temp.mean(dim=0)
        var = embedding_temp.var(dim=0, unbiased=False)
        n = embedding_temp.numel() / embedding_temp.size(1)
        if n > 1:
            var = var * n / (n - 1)
        else:
            var = var 
        test_pred_mean[c]=mean
        test_pred_var[c]=var
    gt_mean_cha= (torch.abs(train_gt_mean-test_gt_mean)).mean(dim=1).numpy()    
    gt_var_cha= (torch.abs(train_gt_var-test_gt_var)).mean(dim=1).numpy()
    pred_mean_cha=(torch.abs(train_pred_mean-test_pred_mean)).mean(dim=1).numpy()
    pred_var_cha=(torch.abs(train_pred_var-test_pred_var)).mean(dim=1).numpy()
    test_gt_var=test_gt_var.mean(dim=1)
    return gt_mean_cha,gt_var_cha,pred_mean_cha,pred_var_cha,test_gt_var