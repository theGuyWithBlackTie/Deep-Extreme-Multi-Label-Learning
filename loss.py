'''
This file is not being used. Kept it for future reference
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss,self).__init__()

    def forward(self, outputs, targets):
        outputs = Variable(outputs, requires_grad=True).to(config.DEVICE)
        targets = Variable(targets, requires_grad=True).to(config.DEVICE)

        resultTensor = torch.sub(outputs, targets)
        rows = resultTensor.shape[0]
        cols = resultTensor.shape[1]
        for eachrow in range(0,rows):
            for eachcol in range(0, cols):
                if resultTensor[eachrow][eachcol] <= 1:
                    resultTensor[eachrow][eachcol].data = torch.mul(torch.square(resultTensor[eachrow][eachcol]), 0.5)
                else:
                    resultTensor[eachrow][eachcol].data = torch.sub(resultTensor[eachrow][eachcol], 0.5)
        
        # Summing each row
        rowSummed           = torch.sum(resultTensor, dim=1)
        allDataPointsSummed = torch.sum(rowSummed, dim=0)
        loss_value          = torch.div(allDataPointsSummed, rows)
        return loss_value

