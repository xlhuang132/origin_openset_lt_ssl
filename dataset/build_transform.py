
from .transforms import *
import torchvision
from .transform.rand_augment import RandAugment
from .transform.transforms import TransformFixMatch,TransformOpenMatch
from .transform.transforms import SimCLRAugmentation

from dataset.transform.transforms import Augmentation,GeneralizedSSLTransform
from dataset.transform.randaugment import RandAugmentMC
from torchvision import transforms


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

# def build_transform(cfg):    
#     transform_train =  Compose([
#         RandomCrop(32),
#         RandomFlip(),
#         ToTensor(),
#     ])

#     transform_val = Compose([
#         CenterCrop(32),
#         ToTensor(),
#     ])
    
   
#     return transform_train,TransformTwice(transform_train),transform_val

 
class TransformFixMatch(object):
    def __init__(self, mean, std, img_size=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        # return self.normalize(weak), self.normalize(strong)

 
def build_simclr_transform(cfg):
    dataset=cfg.DATASET.NAME
    
    resolution = cfg.DATASET.RESOLUTION
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)

    elif dataset == "stl10":
        img_size = (96, 96)  # original image size
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    transform=SimCLRAugmentation(cfg, img_size,norm_params=norm_params, resolution=resolution)
    return TransformTwice(transform)

def build_transform(cfg):
    
    algo_name = cfg.ALGORITHM.NAME 
    # 辅助无标签数据集是否需要强增强  
    
    resolution = cfg.DATASET.RESOLUTION
    
    dataset=cfg.DATASET.NAME
    aug = Augmentation 
     
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)

    elif dataset == "stl10":
        img_size = (96, 96)  # original image size
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
        
    l_train = aug(
            cfg, img_size, strong_aug=cfg.DATASET.TRANSFORM.LABELED_STRONG_AUG, norm_params=norm_params, resolution=resolution
        ) 
    
    if algo_name == "MixMatch":
        # K weak
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution)
                for _ in range(cfg.ALGORITHM.MIXMATCH.NUM_AUG)
            ]
        )

    elif algo_name in ['ReMix']:
        # 1 weak + K strong
        augmentations = [aug(cfg, img_size, norm_params=norm_params, resolution=resolution)]
        for _ in range(cfg.ALGORITHM.REMIXMATCH.NUM_AUG):
            augmentations.append(
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=False
                )
            )
        ul_train = GeneralizedSSLTransform(augmentations)

    elif algo_name == "USADTM":
        # identity + weak + strong
        ul_train = GeneralizedSSLTransform(
            [
                aug(
                    cfg,
                    img_size,
                    norm_params=norm_params,
                    resolution=resolution,
                    flip=False,
                    crop=False
                ),  # identity
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                )  # strong (randaugment)
            ]
        )

    elif algo_name == "PseudoLabel":
        # 1 weak
        ul_train = GeneralizedSSLTransform(
            [aug(cfg, img_size, norm_params=norm_params, resolution=resolution)]
        ) 
    elif algo_name == 'ACR':
        ul_train= GeneralizedSSLTransform(
            [ 
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),  # weak
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                ),  # strong 1(randaugment)
                aug(
                    cfg,
                    img_size,
                    strong_aug=True,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=False
                ),  # strong 2(randaugment)
                
                aug(
                    cfg,
                    img_size,
                    norm_params=norm_params,
                    resolution=resolution,
                    flip=False,
                    crop=False
                ),  # identity
        ]
        )
    else:
        ul_train = GeneralizedSSLTransform(
            [
                aug(cfg, img_size, norm_params=norm_params, resolution=resolution),
                aug(
                    cfg,
                    img_size,
                    strong_aug=cfg.DATASET.TRANSFORM.UNLABELED_STRONG_AUG,
                    norm_params=norm_params,
                    resolution=resolution,
                    ra_first=True
                )
            ]
        )
    eval_aug = Augmentation(
        cfg,
        img_size,
        flip=False,
        crop=False,
        norm_params=norm_params,
        is_train=False,
        resolution=resolution
    )
    return l_train,ul_train,eval_aug