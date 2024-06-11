# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:39:20 2022

@author: marti
"""

import torch
import torch.nn.functional as F

def CrossEntropy2d(input, target, weight=None, reduction='mean'):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, reduction, ignore_index=6)
    elif dim == 4:
        output = input.reshape(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.reshape(-1)  
        return F.cross_entropy(output, target,weight, reduction, ignore_index=6)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))
