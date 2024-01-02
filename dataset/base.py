import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
class BaseNumpyDataset(Dataset):
    """Custom dataset class for classification"""

    def __init__(
        self,
        data_dict: dict,
        image_key: str = "images",
        label_key: str = "labels",
        transforms=None,
        is_ul_unknown=False,
        num_classes=10
    ):
        self.dataset = data_dict
        self.image_key = image_key
        self.label_key = label_key
        self.transforms = transforms
        self.is_ul_unknown = is_ul_unknown
        self.ood_num=0
        self.num_classes=num_classes
        self.total_num=len(self.dataset[self.image_key])
        if not is_ul_unknown:
            self.num_per_cls_list = self._load_num_samples_per_class()
        else:
            self.num_per_cls_list = None

    def __getitem__(self, idx):
        img = self.dataset[self.image_key][idx] #(32, 32, 3)
        label = -1 if self.is_ul_unknown else self.dataset[self.label_key][idx]
        if self.transforms is not None: 
            img = self.transforms(img)                 
        return img, label, idx

    def __len__(self):
        return len(self.dataset[self.image_key])

    # label-to-class quantity
    def _load_num_samples_per_class(self):
        labels = self.dataset[self.label_key]
        classes = range(-1,self.num_classes)

        # classwise_num_samples = dict()
        # for i in classes:
        #     if i==-1:
        #         self.ood_num=len(np.where(labels == i)[0])
        #         continue
        #     classwise_num_samples[i] = len(np.where(labels == i)[0])

        # # in a descending order of classwise count. [(class_idx, count), ...]
        # res = sorted(classwise_num_samples.items(), key=(lambda x: x[1]), reverse=True)
        # return res
        classwise_num_samples = [0]*(len(classes)-1)
        for i in classes:
            if i==-1:
                self.ood_num=len(np.where(labels == i)[0])
                continue
            classwise_num_samples[i] = len(np.where(labels == i)[0])
 
        return np.array(classwise_num_samples)

    def select_dataset(self, indices=None, labels=None, return_transforms=False):
        if indices is None:
            indices = np.array(list([i for i in range(len(self.dataset[self.image_key]))]))
        imgs = self.dataset[self.image_key][indices]

        if not self.is_ul_unknown:
            _labels = self.dataset[self.label_key][indices]
        else:
            _labels = np.array([-1 for _ in range(len(indices))])

        if labels is not None:
            # override specified labels (i.e., pseudo-labels)
            _labels = np.array(labels)

        assert len(_labels) == len(imgs)
        dataset = {self.image_key: imgs, self.label_key: _labels}

        if return_transforms:
            return dataset, self.transforms    
        return dataset
