import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wrn import build_wrn
from .classifier import Classifier

__all__=["WRN_28_2","WRN_28_8","WRN_37_2"]

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2,feature_dim=64,fc2_enable=False, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.encoder=build_wrn(widen_factor,num_classes)
        num_classes = self.encoder.num_classes
        out_features = self.encoder.out_features

        # classifier
        self.fc = Classifier(out_features, num_classes)
        self.fc2_enable=fc2_enable
        if self.fc2_enable:
            self.fc2 = Classifier(out_features, 1) 
        # misc
        self.num_classes = num_classes
        self.out_features = out_features 
    
    def forward(self, x,return_encoding=False,classifier=False): 
        if classifier:
            if self.fc2_enable:
                return self.fc(x),self.fc2(x)
            return self.fc(x)        
        encoding = self.encoder(x)
        if return_encoding:
            return encoding 
        out=self.fc(encoding)
        if self.fc2_enable: 
            out2=self.fc2(encoding) 
            return out,out2   
        return out
        # if return_encoding:
        #     return  F.normalize(out , dim=-1) ,encoding
        # return F.normalize(out , dim=-1) 
    
def WRN_28_2(cfg):
    num_classes=cfg.DATASET.NUM_CLASSES 
    fc2_enable=cfg.MODEL.DUAL_HEAD_ENABLE
    return WideResNet(num_classes,depth=28,widen_factor=2,fc2_enable=fc2_enable)

def WRN_28_8(cfg):
    num_classes=cfg.DATASET.NUM_CLASSES
    fc2_enable=cfg.MODEL.DUAL_HEAD_ENABLE
    return WideResNet(num_classes,depth=28,widen_factor=8,fc2_enable=fc2_enable)

def WRN_37_2(cfg):    
    num_classes=cfg.DATASET.NUM_CLASSES
    fc2_enable=cfg.MODEL.DUAL_HEAD_ENABLE
    return WideResNet(num_classes,depth=37,widen_factor=2,fc2_enable=fc2_enable)
  