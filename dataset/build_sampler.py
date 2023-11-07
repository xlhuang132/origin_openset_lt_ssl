
from .class_aware_sampler import ClassAwareSampler 
from .random_sampler import RandomSampler
from .class_balanced_sampler import ClassBalancedSampler
from .class_reversed_sampler import ClassReversedSampler
 

def build_sampler(cfg,dataset,sampler_type="RandomSampler",total_samples=None):
    assert sampler_type!=None and total_samples!=None
    if sampler_type == "RandomSampler": 
        sampler = RandomSampler(dataset,total_samples=total_samples) 
        
    elif sampler_type == "ClassAwareSampler":
        sampler = ClassAwareSampler(dataset, total_samples)
        print(
                "ClassAwareSampler is enabled.  "
                "per_class probabilities: {}".format(
                    ", ".join(["{:.4f}".format(v) for v in sampler.per_cls_prob])
                )
            )
    elif sampler_type == "ClassBalancedSampler": # 每个类的采样概率一样
        sampler = ClassBalancedSampler(dataset, total_samples)
        print(
                "ClassBalancedSampler is enabled.  " 
            )
    elif sampler_type == "ClassReversedSampler": # 逆类概率采样
        sampler = ClassReversedSampler(dataset, total_samples)
        print(
                "ClassReversedSampler is enabled.  " 
            )
    else:
        raise ValueError
    
    
    return sampler