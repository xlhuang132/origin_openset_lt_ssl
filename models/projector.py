import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, cfg=None,model_name=''):
        super(Projector, self).__init__()
        if cfg is not None:
            model_name=cfg.MODEL.NAME
        if model_name == 'WRN_28_2':
            self.linear_1 = nn.Linear(128, 128)
            self.linear_2 = nn.Linear(128, 128)
        elif model_name == 'Resnet34':
            self.linear_1 = nn.Linear(512, 128)
            self.linear_2 = nn.Linear(128, 128)
        elif in_feature_dim==2048:
            self.linear_1 = nn.Linear(2048, 2048)
            self.linear_2 = nn.Linear(2048, 2048)
        else:
            self.linear_1 = nn.Linear(512, 128)
            self.linear_2 = nn.Linear(128, 128)
            

    def forward(self, x, internal_output_list=False):
            
        #output_list = []

        output = self.linear_1(x)
        output = F.relu(output)
        #output_list.append(output)

        output = F.normalize(self.linear_2(output),dim=-1)

        #output_list.append(output)


        return output 