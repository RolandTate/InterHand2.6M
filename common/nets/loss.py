# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from main.config import cfg
import math

class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        loss = (joint_out - joint_gt)**2 * joint_valid[:,:,None,None,None]
        return loss

class MPJPELoss(nn.Module):
    def __int__(self):
        super(MPJPELoss).__int__()

    def forward(self, joint_out, joint_gt):
        loss = torch.sqrt(torch.sum((joint_out - joint_gt)**2, dim=2))

        return loss


class JointCoordLoss(nn.Module):
    def __ini__(self):
        super(JointCoordLoss, self).__init__()

    def forward(self, joint_out, joint_gt):
        loss = torch.abs(joint_out - joint_gt)
        return loss

class HandTypeLoss(nn.Module):
    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt, reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss

class RelRootDepthLoss(nn.Module):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = torch.abs(root_depth_out - root_depth_gt) * root_valid
        return loss

class DiscriminateLoss(nn.Module):
    def __int__(self):
        super(DiscriminateLoss, self).__int__()

    def forward(self, dis_feature):
        loss1 = F.binary_cross_entropy(dis_feature[0], torch.ones_like(dis_feature[0]).cuda(), reduction='none')
        loss2 = F.binary_cross_entropy(dis_feature[1], torch.zeros_like(dis_feature[1]).cuda(), reduction='none')
        loss = loss1 + loss2

        return loss