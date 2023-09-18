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
from nets.layer import make_linear_layers, make_conv_layers, make_deconv_layers, make_upsample_layers
from nets.resnet import ResNetBackbone
import math

from common.nets.layer import make_GAT_layers, BGAT, GATBlock


class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.resnet = ResNetBackbone(cfg.resnet_type)
    
    def init_weights(self):
        self.resnet.init_weights()

    def forward(self, img):
        img_feat = self.resnet(img)
        return img_feat

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num # single hand
        
        self.joint_deconv_1 = make_deconv_layers([2048,256,256,256])
        self.joint_conv_1 = make_conv_layers([256,self.joint_num*cfg.output_hm_shape[0]],kernel=1,stride=1,padding=0,bnrelu_final=False)
        self.joint_deconv_2 = make_deconv_layers([2048,256,256,256])
        self.joint_conv_2 = make_conv_layers([256,self.joint_num*cfg.output_hm_shape[0]],kernel=1,stride=1,padding=0,bnrelu_final=False)
        
        self.root_fc = make_linear_layers([2048,512,cfg.output_root_hm_shape],relu_final=False)
        self.hand_fc = make_linear_layers([2048,512,2],relu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(cfg.output_root_hm_shape).float().cuda()[None,:]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, img_feat):
        # img_feat.shape: torch.Size([16, 2048, 8, 8])
        joint_img_feat_1 = self.joint_deconv_1(img_feat)
        # joint_img_feat_1.shape: torch.Size([16, 256, 64, 64])
        joint_heatmap3d_1 = self.joint_conv_1(joint_img_feat_1).view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        # self.joint_conv_1(joint_img_feat_1).shape: torch.Size([16, 1344, 64, 64])
        # joint_heatmap3d_1.shape: torch.Size([16, 21, 64, 64, 64])
        joint_img_feat_2 = self.joint_deconv_2(img_feat)
        # joint_img_feat_2.shape: torch.Size([16, 256, 64, 64])
        joint_heatmap3d_2 = self.joint_conv_2(joint_img_feat_2).view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        # self.joint_conv_2(joint_img_feat_2).shape: torch.Size([16, 1344, 64, 64])
        # joint_heatmap3d_2.shape: torch.Size([16, 21, 64, 64, 64])
        joint_heatmap3d = torch.cat((joint_heatmap3d_1, joint_heatmap3d_2),1)
        # joint_heatmap3d.shape: torch.Size([16, 42, 64, 64, 64])
        
        img_feat_gap = F.avg_pool2d(img_feat, (img_feat.shape[2],img_feat.shape[3])).view(-1,2048)
        # img_feat_gap.shape: torch.Size([16, 2048])
        root_heatmap1d = self.root_fc(img_feat_gap)
        # root_heatmap1d.shape: torch.Size([16, 64])
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1,1)
        # root_depth.shape: torch.Size([16, 1])
        hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))
        # hand_type.shape: torch.Size([16, 2])

        return joint_heatmap3d, root_depth, hand_type


class GAT_PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(GAT_PoseNet, self).__init__()
        self.joint_num = joint_num  # single hand

        # self.joint_deconv_1 = make_deconv_layers([2048, 512, 128, 42])
        # self.joint_full_GAT_1 = BGAT(64*64, 2048, 1024)
        # self.joint_hand_GAT_1 = BGAT(1024, 512, 256)
        # self.joint_linear_1 = make_linear_layers([256, 64, 3])

        self.joint_deconv_1 = make_deconv_layers([2048, 512, 128, 21])
        # self.joint_full_GAT_1 = BGAT(64 * 64, 2048, 1024)
        # self.joint_hand_GAT_1 = BGAT(1024, 512, 256)
        # self.joint_linear_1 = make_linear_layers([256, 64, 3])
        #
        self.joint_deconv_2 = make_deconv_layers([2048, 512, 128, 21])
        # self.joint_full_GAT_2 = BGAT(64 * 64, 2048, 1024)
        # self.joint_hand_GAT_2 = BGAT(1024, 512, 256)
        # self.joint_linear_2 = make_linear_layers([256, 64, 3])

        self.GATBlock1 = GATBlock(4096, 2048, 1024)
        self.GATBlock2 = GATBlock(1024, 512, 256)
        self.joint_linear = make_linear_layers([256, 64, 3])

        self.root_fc = make_linear_layers([2048, 512, cfg.output_root_hm_shape], relu_final=False)
        self.hand_fc = make_linear_layers([2048, 512, 2], relu_final=False)

        adjs = self.build_hand_adj()
        self.single_adj = adjs[0].cuda()
        self.cross_adj = adjs[1].cuda()
        # self.hand_adj = self.build_hand_adj().cuda()
        # self.fuc_adj = torch.ones(21, 21).cuda()

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(cfg.output_root_hm_shape).float().cuda()[None, :]
        coord = accu.sum(dim=1)
        return coord


    def build_hand_adj(self):
        num_nodes = 21
        single_adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        cross_adj_matrix = np.ones((num_nodes*2, num_nodes*2), dtype=int)

        # # 定义节点之间的连接关系
        single_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 手指1
            (0, 5), (5, 6), (6, 7), (7, 8),  # 手指2
            (0, 9), (9, 10), (10, 11), (11, 12),  # 手指3
            (0, 13), (13, 14), (14, 15), (15, 16),  # 手指4
            (0, 17), (17, 18), (18, 19), (19, 20)  # 手指5
        ]

        # 定义节点之间的连接关系
        cross_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 手指1
            (0, 5), (5, 6), (6, 7), (7, 8),  # 手指2
            (0, 9), (9, 10), (10, 11), (11, 12),  # 手指3
            (0, 13), (13, 14), (14, 15), (15, 16),  # 手指4
            (0, 17), (17, 18), (18, 19), (19, 20),  # 手指5
            (21, 22), (22, 23), (23, 24), (24, 25),  # 手指6
            (21, 26), (26, 27), (27, 28), (28, 29),  # 手指7
            (21, 30), (30, 31), (31, 32), (32, 33),  # 手指3
            (21, 34), (34, 35), (35, 36), (36, 37),  # 手指4
            (21, 38), (38, 39), (39, 40), (40, 41)  # 手指10
        ]

        # 根据连接关系更新邻接矩阵
        for connection in single_connections:
            i, j = connection
            single_adj_matrix[i][j] = 1
            single_adj_matrix[j][i] = 1

        for connection in cross_connections:
            i, j = connection
            cross_adj_matrix[i][j] = 0
            cross_adj_matrix[j][i] = 0

        # print(adj_matrix)
        return torch.tensor(single_adj_matrix), torch.tensor(cross_adj_matrix)

    def forward(self, img_feat):
        # joint_img_feat_1 = self.joint_deconv_1(img_feat)
        # joint_img_feat_1 = joint_img_feat_1.clone().view(joint_img_feat_1.size(0), joint_img_feat_1.size(1), -1)
        # # print(f'joint_img_feat_1.shape: {joint_img_feat_1.shape}')
        # joint_coord3d_1 = self.joint_full_GAT_1(joint_img_feat_1, self.fuc_adj)
        # joint_coord3d_1 = self.joint_hand_GAT_1(joint_coord3d_1, self.hand_adj)
        # joint_coord3d_1 = self.joint_linear_1(joint_coord3d_1)

        joint_img_feat_1 = self.joint_deconv_1(img_feat)  #16, 3, 256,256->16，2048，8，8->16, 21, 64,64->16,21,64x64
        # 16, 3, 256,256->16，2048，8，8->16,21x64,64,64->16,21,64,64,64
        joint_img_feat_1 = joint_img_feat_1.clone().view(joint_img_feat_1.size(0), joint_img_feat_1.size(1), -1)
        # joint_coord3d_1 = self.joint_full_GAT_1(joint_img_feat_1, self.fuc_adj)
        # joint_coord3d_1 = self.joint_hand_GAT_1(joint_coord3d_1, self.hand_adj)
        # joint_coord3d_1 = self.joint_linear_1(joint_coord3d_1)

        joint_img_feat_2 = self.joint_deconv_2(img_feat)
        joint_img_feat_2 = joint_img_feat_2.clone().view(joint_img_feat_2.size(0), joint_img_feat_2.size(1), -1)
        # joint_coord3d_2 = self.joint_full_GAT_2(joint_img_feat_2, self.fuc_adj)
        # joint_coord3d_2 = self.joint_hand_GAT_2(joint_coord3d_2, self.hand_adj)
        # joint_coord3d_2 = self.joint_linear_2(joint_coord3d_2)

        cross_joint_coord3d = self.GATBlock1(joint_img_feat_1, joint_img_feat_2, self.single_adj, self.cross_adj)
        joint_coord3d_1, cross_joint_coord3d_2 = torch.chunk(cross_joint_coord3d, 2, dim=1)
        joint_coord3d = self.GATBlock2(joint_coord3d_1, cross_joint_coord3d_2, self.single_adj, self.cross_adj)
        joint_coord3d = self.joint_linear(joint_coord3d)

        # joint_coord3d = torch.cat([joint_coord3d_1, joint_coord3d_2], 1)
        # joint_coord3d = joint_coord3d_1

        img_feat_gap = F.avg_pool2d(img_feat, (img_feat.shape[2], img_feat.shape[3])).view(-1, 2048)
        root_heatmap1d = self.root_fc(img_feat_gap)
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1, 1)
        hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))

        return joint_coord3d, root_depth, hand_type
