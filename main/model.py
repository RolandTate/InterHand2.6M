# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.nets.module import BackboneNet, PoseNet, EasyBackboneNet
from common.nets.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss, DiscriminateLoss

from common.nets.loss import JointCoordLoss, MPJPELoss
from common.nets.module import GAT_PoseNet
from common.nets.pyramid_vig import pvig_s_224_gelu
from config import cfg
import math


class Model(nn.Module):
    def __init__(self, backbone_net, pose_net):
        super(Model, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.pose_net = pose_net

        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()

        self.mpjpe_loss = MPJPELoss()

    def render_gaussian_heatmap(self, joint_coord):
        x = torch.arange(cfg.output_hm_shape[2])  # 64
        y = torch.arange(cfg.output_hm_shape[1])  # 64
        z = torch.arange(cfg.output_hm_shape[0])  # 64
        zz, yy, xx = torch.meshgrid(z, y, x)
        xx = xx[None, None, :, :, :].cuda().float();
        yy = yy[None, None, :, :, :].cuda().float();
        zz = zz[None, None, :, :, :].cuda().float();
        # xx.shape: {[1, 1, 64, 64, 64]}
        x = joint_coord[:, :, 0, None, None, None];
        y = joint_coord[:, :, 1, None, None, None];
        z = joint_coord[:, :, 2, None, None, None];
        # x.shape: {16, 42, 1, 1, 1}
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
        heatmap = heatmap * 255
        return heatmap

    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']  # input_img.shape: torch.Size([16, 3, 256, 256])
        batch_size = input_img.shape[0]
        img_feat = self.backbone_net(input_img)  # img_feat.shape: torch.Size([16, 2048, 8, 8])
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(img_feat)
        # joint_heatmap_out.shape: torch.Size([16, 42, 64, 64, 64])
        # rel_root_depth_out.shape: torch.Size([16, 1])
        # hand_type.shape: torch.Size([16, 2])

        if mode == 'train':
            target_joint_heatmap = self.render_gaussian_heatmap(targets['joint_coord'])
            # targets['joint_coord'].shape: torch.Size([16, 42, 3])
            # target_joint_heatmap.shape: torch.Size([16, 42, 64, 64, 64])

            loss = {}
            # loss['joint_heatmap'] = self.mpjpe_loss(joint_coord_out, targets['joint_coord'])
            loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap,
                                                            meta_info['joint_valid'])
            loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, targets['rel_root_depth'],
                                                              meta_info['root_valid'])
            loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'], meta_info['hand_type_valid'])
            return loss
        elif mode == 'test':
            out = {}
            val_z, idx_z = torch.max(joint_heatmap_out, 2)
            # val_z.shape: torch.Size([16, 42, 64, 64]), idx_z.shape: torch.Size([1, 42, 64, 64])
            val_zy, idx_zy = torch.max(val_z, 2)
            # val_zy.shape: torch.Size([16, 42, 64]), idx_zy.shape: torch.Size([1, 42, 64])
            val_zyx, joint_x = torch.max(val_zy, 2)
            # val_zyx.shape: torch.Size([16, 42]), joint_x.shape: torch.Size([1, 42])
            joint_x = joint_x[:, :, None]
            # joint_x.shape: torch.Size([16, 42, 1])
            joint_y = torch.gather(idx_zy, 2, joint_x)
            # joint_y.shape: torch.Size([16, 42, 1])
            joint_z = torch.gather(idx_z, 2, joint_y[:, :, :, None].repeat(1, 1, 1, cfg.output_hm_shape[1]))[:, :, 0, :]
            # joint_z.shape: torch.Size([16, 42, 64])
            joint_z = torch.gather(joint_z, 2, joint_x)
            # joint_z.shape: torch.Size([16, 42, 1])
            joint_coord_out = torch.cat((joint_x, joint_y, joint_z), 2).float()
            # joint_coord_out.shape: torch.Size([16, 42, 3])
            out['joint_coord'] = joint_coord_out
            out['rel_root_depth'] = rel_root_depth_out
            out['hand_type'] = hand_type
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            return out


class GNN_Model(nn.Module):
    def __init__(self, backbone_net, gat_pose_net):
        super(GNN_Model, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.gat_pose_net = gat_pose_net

        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.joint_coord_loss = JointCoordLoss()
        self.mpjpe_loss = MPJPELoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()
        self.dis_loss = DiscriminateLoss()


    def render_gaussian_heatmap(self, joint_coord):
        x = torch.arange(cfg.output_hm_shape[2])  # 64
        y = torch.arange(cfg.output_hm_shape[1])  # 64
        z = torch.arange(cfg.output_hm_shape[0])  # 64
        zz, yy, xx = torch.meshgrid(z, y, x)
        xx = xx[None, None, :, :, :].cuda().float();
        yy = yy[None, None, :, :, :].cuda().float();
        zz = zz[None, None, :, :, :].cuda().float();
        # xx.shape: {[1, 1, 64, 64, 64]}
        x = joint_coord[:, :, 0, None, None, None];
        y = joint_coord[:, :, 1, None, None, None];
        z = joint_coord[:, :, 2, None, None, None];
        # x.shape: {16, 42, 1, 1, 1}
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
        heatmap = heatmap * 255
        return heatmap


    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']
        img_feat = self.backbone_net(input_img)
        joint_coord3d, rel_root_depth_out, hand_type, dis_feature= self.gat_pose_net(img_feat)  # 16，21， 3

        if mode == 'train':
            # pre_joint_heatmap = self.render_gaussian_heatmap(joint_coord3d)
            # target_joint_heatmap = self.render_gaussian_heatmap(targets['joint_coord'])

            loss = {}
            loss['joint_coord'] = 0.01 * self.joint_coord_loss(joint_coord3d, targets['joint_coord'])
            # loss['joint_heatmap'] = self.joint_heatmap_loss(pre_joint_heatmap, target_joint_heatmap,
            #                                                 meta_info['joint_valid'])
            loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, targets['rel_root_depth'],
                                                              meta_info['root_valid'])
            loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'], meta_info['hand_type_valid'])

            loss['discriminate'] = self.dis_loss(dis_feature)
            return loss
        elif mode == 'test':
            out = {}
            out['joint_coord'] = joint_coord3d
            out['rel_root_depth'] = rel_root_depth_out
            out['hand_type'] = hand_type
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode, joint_num):
    backbone_net = BackboneNet()
    pose_net = PoseNet(joint_num)

    if mode == 'train':
        backbone_net.init_weights()
        pose_net.apply(init_weights)

    model = Model(backbone_net, pose_net)
    return model


def get_model_GNN(mode, joint_num):
    backbone_net = BackboneNet()
    gat_pose_net = GAT_PoseNet(joint_num)

    if mode == 'train':
        backbone_net.init_weights()

    model = GNN_Model(backbone_net, gat_pose_net)
    return model
