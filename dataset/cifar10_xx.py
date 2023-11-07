import numpy as np 
from .utils import train_val_split
import torchvision 
from PIL import Image 
from .build_transform import  build_simclr_transform
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
     
 
def get_cifar10_test_dataset(root,transform_val=None):
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)
    return test_dataset

def get_cifar10_for_ood_detection(root, out_dataset, start_label=0, 
                  transform_val=None,
                 download=True,cfg=None,logger=None):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    
    train_labeled_idxs, train_unlabeled_idxs, val_idxs,l_num_per_cls_list,_ = train_val_split(base_dataset.targets,cfg)
 
    # 所有的训练数据连接在一起
    train_dataset = CIFAR10_cocnat(root, train_labeled_idxs, train_unlabeled_idxs, start_label, OOD_path=f'./data/{out_dataset}', train=True,ood_train=False, transform=transform_val)

    if logger:
        logger.info (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #OOD: {len(train_unlabeled_dataset.OOD)} #Val: {len(val_idxs)}")
        logger.info("*"*70)
    else:
        print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #OOD: {len(train_dataset.OOD)} #Val: {len(val_idxs)}")
        print("*"*70)
    return train_dataset 
 

def get_cifar10(root, out_dataset, start_label=0,ood_ratio=0, 
                 transform_train=None, transform_val=None,test_mode=False,
                 transform_train_ul=None,
                 download=True,cfg=None,logger=None):
    if logger:
        logger.info("*"*70)
    else:print("*"*70)
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    
    train_labeled_idxs, train_unlabeled_idxs, val_idxs,l_num_per_cls_list,ood_num = train_val_split(base_dataset.targets,cfg)
    
    if test_mode:
        train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_val,num_per_cls_list=l_num_per_cls_list)
        if cfg.ALGORITHM.NAME in["Ours","OODMix","CReST"]:
            train_unlabeled_dataset = CIFAR10_unlabeled(root, 
                train_unlabeled_idxs, OOD_path=f'./data/{out_dataset}',
                 ood_num=ood_num,train=True,
                transform=transform_val,return_index=True)
        
        else:
            train_unlabeled_dataset = CIFAR10_unlabeled(root, 
                train_unlabeled_idxs, 
                OOD_path=f'./data/{out_dataset}', 
                ood_num=ood_num,train=True, 
                transform=transform_val)
        
        if cfg.ALGORITHM.PRE_TRAIN.SimCLR.ENABLE:
            transform_pre=build_simclr_transform(cfg)
            pre_train_dataset = CIFAR10_cocnat_pair(
                root, train_labeled_idxs, train_unlabeled_idxs,
                start_label,transform=transform_val, 
                OOD_path=f'./data/{out_dataset}',
                train=True,ood_num=ood_num)

        train_dataset = CIFAR10_cocnat(
            root, train_labeled_idxs, train_unlabeled_idxs, 
            start_label, OOD_path=f'./data/{out_dataset}', 
            train=True,ood_num=ood_num, transform=transform_val)

        val_dataset = CIFAR10_labeled(
            root, val_idxs, train=True, 
            transform=transform_val, 
            download=True)
        test_dataset = CIFAR10_labeled(
            root, train=False, 
            transform=transform_val, 
            download=True)
    else:
        train_labeled_dataset = CIFAR10_labeled(
            root, train_labeled_idxs, train=True, 
            transform=transform_train,
            num_per_cls_list=l_num_per_cls_list)
      
        train_unlabeled_dataset = CIFAR10_unlabeled(
            root, train_unlabeled_idxs, 
            OOD_path=f'./data/{out_dataset}', ood_num=ood_num,train=True,
            transform=transform_train_ul,
            return_index=cfg.DATASET.UNLABELED_DATASET_RETURN_INDEX)
        
      
        if cfg.ALGORITHM.PRE_TRAIN.SimCLR.ENABLE:
            transform_pre=build_simclr_transform(cfg)
            pre_train_dataset = CIFAR10_cocnat_pair(root, train_labeled_idxs, train_unlabeled_idxs,
                                                    start_label,transform=transform_pre, 
                                                    OOD_path=f'./data/{out_dataset}',
                                                    train=True,ood_num=ood_num)
    
        # 所有的数据concat数据集, 为了mtcf的训练
        train_dataset = CIFAR10_cocnat(root, train_labeled_idxs, train_unlabeled_idxs, 
                                       start_label, OOD_path=f'./data/{out_dataset}', 
                                       train=True,ood_num=ood_num, transform=transform_train, 
                                       )

        val_dataset = CIFAR10_labeled(
            root, val_idxs, train=True, 
            transform=transform_val, 
            download=True)
        test_dataset = CIFAR10_labeled(
            root, train=False, 
            transform=transform_val, 
            download=True)
    if logger:
        logger.info (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #OOD: {len(train_unlabeled_dataset.OOD)} #Val: {len(val_idxs)}")
        logger.info("*"*70)
    else:
        print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #OOD: {len(train_unlabeled_dataset.OOD)} #Val: {len(val_idxs)}")
        print("*"*70)
    if cfg.ALGORITHM.PRE_TRAIN.SimCLR.ENABLE:
        return train_labeled_dataset, train_unlabeled_dataset, train_dataset, val_dataset, test_dataset,pre_train_dataset
    else:
        return train_labeled_dataset, train_unlabeled_dataset, train_dataset, val_dataset, test_dataset
 
 
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def pad(x, border=4):
    return np.pad(x, [(0,0), (border, border), (border, border), (0, 0)])

def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False,num_per_cls_list=None):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(pad(self.data)))
        self.num_per_cls_list=num_per_cls_list
        
    def select_dataset(self, indices=None, labels=None, return_transform=False):
        if indices is None:
            indices = list(range(len(self.data)))

        imgs = self.data[indices]

        _labels = self.targets[indices] 

        if labels is not None:
            # override specified labels (i.e., pseudo-labels)
            _labels = np.array(labels)

        assert len(_labels) == len(imgs)
        dataset = {"images": imgs, "labels": _labels}

        if return_transform:
            return dataset, self.transform    
        return dataset

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target,index

class CIFAR10_unlabeled(CIFAR10_labeled):
    def get_ood_num(self,ratio,total_ood_num,unlabel_num):
        n_ood=int(ratio*unlabel_num)
        if n_ood>len(OOD):
            print('OOD samples are not enough to meet the setting of ratio{}'.format(ratio))
        return min(n_ood,len(OOD))

    def __init__(self, root, indexs, OOD_path, train=True,
                 transform=None, target_transform=None,ood_num=None,  return_index=False,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.return_index=return_index
        if OOD_path == './data/none':
             self.OOD = []
        else:
            self.OOD = np.load(OOD_path+'.npy')
            if ood_num!=None:
                self.ood_num=ood_num
            else:
                self.ood_num=len(self.OOD)
            if self.ood_num>len(self.OOD):
                self.ood_num=len(self.OOD)
                print('OOD samples are not enough to meet the setting of ood num{}'.format(ood_num))
         
            self.OOD=self.OOD[:self.ood_num]
            self.OOD = transpose(normalize(pad(self.OOD))) 
        self.targets = np.array(self.targets.tolist() + [-1]*len(self.OOD))
        print("-1 num:{}".format(np.sum(self.targets==-1)))
        print("")

    def __len__(self):
        return len(self.data) + len(self.OOD)

    
    def select_dataset(self, indices=None, labels=None, return_transform=False):
        if indices is None:
            indices = list(range(len(self.data)))
        imgs=[]
        _labels=[]
        for idx in indices:
            data=self.get_data(idx) 
            imgs.append(data[0])
            _labels.append(data[1])

        if labels is not None:
            # override specified labels (i.e., pseudo-labels)
            _labels = np.array(labels)

        assert len(_labels) == len(imgs)
        dataset = {"images": np.array(imgs), "labels": _labels}

        if return_transform:
            return dataset, self.transform    
        return dataset
    
    def get_data(self, index):
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.OOD[index - len(self.data)], self.targets[index]
        return img, target
    
    def __getitem__(self, index):
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.OOD[index - len(self.data)], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
    
class CIFAR10_cocnat(torchvision.datasets.CIFAR10):

    def __init__(self, root, labeled_indexs, unlabeled_indexs, start_label, OOD_path,ood_num=None, train=True,
                 transform=None, target_transform=None,ood_train=True,
                 download=False):
        super(CIFAR10_cocnat, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)

        self.data_x = self.data[labeled_indexs]
        self.data_x = transpose(normalize(pad(self.data_x)))
        self.targets_x = np.array(self.targets)[labeled_indexs]

        self.data_u = self.data[unlabeled_indexs]
        self.data_u = transpose(normalize(pad(self.data_u)))
        self.targets_u = np.array(self.targets)[unlabeled_indexs]
        
        if OOD_path == './data/none':
             self.OOD = []
             self.ood_num=0
        else:
            self.OOD = np.load(OOD_path+'.npy')
            
            if ood_num!=None:
                self.ood_num=min(len(self.OOD), ood_num)
            else:
                self.ood_num=len(self.OOD)
            self.OOD = self.OOD[:self.ood_num]
            self.OOD = transpose(normalize(pad(self.OOD)))
        #
        self.soft_labels = np.zeros((len(self.data_x)+len(self.data_u)+len(self.OOD)), dtype=np.float32)
            
        if ood_train:
            for idx in range(len(self.data_x)+len(self.data_u)+len(self.OOD)):
                if idx < len(self.data_x):
                    self.soft_labels[idx] = 1.0
                else:
                    self.soft_labels[idx] = start_label
        else:
            for idx in range(len(self.data_x)+len(self.data_u)+len(self.OOD)):
                if idx < len(self.data_x)+len(self.data_u):
                    self.soft_labels[idx] = 1.0
                else:
                    self.soft_labels[idx] = start_label
        self.prediction = np.zeros((len(self.data_x)+len(self.data_u)+len(self.OOD), 10), dtype=np.float32)
        self.prediction[:len(self.data_x),:] = 1.0
        self.count = 0

    def __len__(self):
        return len(self.data_x) + len(self.data_u) + len(self.OOD)

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[len(self.data_x):, idx] = results[len(self.data_x):]

        if self.count >= 10:
            self.soft_labels = self.prediction.mean(axis=1)

    def __getitem__(self, index):
        if index < len(self.data_x):
            img, target = self.data_x[index], self.targets_x[index]

        elif index < len(self.data_x) + len(self.data_u):
            img, target = self.data_u[index-len(self.data_x)], self.targets_u[index-len(self.data_x)]

        else:
            img, target = self.OOD[index - len(self.data_x) - len(self.data_u)], -1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.soft_labels[index], index
    
 
class CIFAR10_cocnat_pair(torchvision.datasets.CIFAR10):

    def __init__(self, root, labeled_indexs, unlabeled_indexs, start_label, OOD_path,ood_num=None, train=True,
                 transform=None, target_transform=None,ood_train=True,
                 download=False):
        super(CIFAR10_cocnat_pair, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)

        self.data_x = self.data[labeled_indexs]
        # self.data_x = transpose(normalize(pad(self.data_x)))
        self.targets_x = np.array(self.targets)[labeled_indexs]

        self.data_u = self.data[unlabeled_indexs]
        # self.data_u = transpose(normalize(pad(self.data_u)))
        self.targets_u = np.array(self.targets)[unlabeled_indexs]
        
        if OOD_path == './data/none':
             self.OOD = []
             self.ood_num=0
        else:
            self.OOD = np.load(OOD_path+'.npy')
            
            if ood_num!=None:
                self.ood_num=min(len(self.OOD), ood_num)
            else:
                self.ood_num=len(self.OOD)
            self.OOD = self.OOD[:self.ood_num]
            # self.OOD = transpose(normalize(pad(self.OOD)))
        #
        self.soft_labels = np.zeros((len(self.data_x)+len(self.data_u)+len(self.OOD)), dtype=np.float32)
            
        if ood_train:
            for idx in range(len(self.data_x)+len(self.data_u)+len(self.OOD)):
                if idx < len(self.data_x):
                    self.soft_labels[idx] = 1.0
                else:
                    self.soft_labels[idx] = start_label
        else:
            for idx in range(len(self.data_x)+len(self.data_u)+len(self.OOD)):
                if idx < len(self.data_x)+len(self.data_u):
                    self.soft_labels[idx] = 1.0
                else:
                    self.soft_labels[idx] = start_label
        self.prediction = np.zeros((len(self.data_x)+len(self.data_u)+len(self.OOD), 10), dtype=np.float32)
        self.prediction[:len(self.data_x),:] = 1.0
        self.count = 0
        self.transform=transform

    def __len__(self):
        return len(self.data_x) + len(self.data_u) + len(self.OOD)

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[len(self.data_x):, idx] = results[len(self.data_x):]

        if self.count >= 10:
            self.soft_labels = self.prediction.mean(axis=1)

    def __getitem__(self, index):
        if index < len(self.data_x):
            img, target = self.data_x[index], self.targets_x[index]

        elif index < len(self.data_x) + len(self.data_u):
            img, target = self.data_u[index-len(self.data_x)], self.targets_u[index-len(self.data_x)]

        else:
            img, target = self.OOD[index - len(self.data_x) - len(self.data_u)], -1
        
        img = Image.fromarray(img)
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
 
        
        return pos_1, pos_2, target

    
   