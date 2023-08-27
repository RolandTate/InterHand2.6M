# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch_geometric.nn import GATConv, BatchNorm

from common.gcn_lib import Grapher, act_layer
from common.nets.layer import make_linear_layers


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),  # [(输入-卷积核+2*P/步长]+1,(input-1)/2 +1->input/2
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.deConv = nn.Sequential(
            nn.ConvTranspose2d(  # (input-1)*stride+ output_padding -2*padding+kernel_size, (input-1)*2+2=input*2
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.deConv(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k  # 邻居节点数，默认为9
        act = opt.act  # 激活层类型，可选 {relu, prelu, leakyrelu, gelu, hswish}
        norm = opt.norm  # 归一化方式，可选 {batch, instance}
        bias = opt.bias  # 卷积层是否使用偏置
        epsilon = opt.epsilon  # gcn的随机采样率
        stochastic = opt.use_stochastic  # gcn的随机性
        conv = opt.conv  # 图卷积层类型，可选 {edge, mr}
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        
        blocks = opt.blocks  # 各层的block个数
        self.n_blocks = sum(blocks)
        channels = opt.channels  # 各层的通道数
        reduce_ratios = [4, 2, 1, 1, 1, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        
        self.stem = Stem(out_dim=channels[0], act=act)
        # self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        # HW = 224 // 4 * 224 // 4
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 256 // 4, 256 // 4))
        HW = 256 // 4 * 256 // 4

        self.backbone = nn.ModuleList([])
        self.pose_predict1 = nn.ModuleList([])
        self.pose_predict2 = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):  # blocks: [2, 2, 6, 2, 1, 1, 1]
            # channels = [256, 512, 1024, 2048, 1024, 512, 1344]
            if i == 0:
                for j in range(blocks[i]):
                    self.backbone += [
                        Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                            FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                            )]
                    idx += 1
            elif (i > 0) and (i < 4):
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                print(f'channels[i - 1], channels[i]: {channels[i - 1], channels[i]}')
                HW = HW // 4
                for j in range(blocks[i]):
                    self.backbone += [
                        Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                            FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                            )]
                    idx += 1
            else:
                self.pose_predict1.append(Upsample(channels[i - 1], channels[i]))
                print(f'channels[i - 1], channels[i]: {channels[i - 1], channels[i]}')
                self.pose_predict2.append(Upsample(channels[i - 1], channels[i]))
                HW = HW * 4
                for j in range(blocks[i]):
                    self.pose_predict1 += [
                        Seq(Grapher(channels[i], 9, min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, 1, n=HW, drop_path=0.0,
                                    relative_pos=True),
                            FFN(channels[i], channels[i] * 4, act=act, drop_path=0.0)
                            )]
                    self.pose_predict2 += [
                        Seq(Grapher(channels[i], 9, min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, 1, n=HW, drop_path=0.0,
                                    relative_pos=True),
                            FFN(channels[i], channels[i] * 4, act=act, drop_path=0.0)
                            )]
                    idx += 1

        self.backbone = Seq(*self.backbone)
        self.pose_predict1 = Seq(*self.pose_predict1)
        self.pose_predict2 = Seq(*self.pose_predict2)

        self.root_fc = make_linear_layers([2048, 512, 64], relu_final=False)
        self.hand_fc = make_linear_layers([2048, 512, 2], relu_final=False)

        # self.GATConv1 = []
        # self.GATConv1.append(GATConv(64*64*64, 1024))
        # self.GATConv1.append(BatchNorm(1024))
        #
        # self.GATConv2 = GATConv(1024, 1024)
        # self.bn2 = BatchNorm(1024)

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(64).float().cuda()[None,:]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        x1, x2 = torch.tensor(x), torch.tensor(x)
        for i in range(len(self.pose_predict1)):
            x1 = self.pose_predict1[i](x1)
        x1 = x1.view(-1, 21, 64, 64, 64)
        for i in range(len(self.pose_predict2)):
            x2 = self.pose_predict1[i](x2)
        x2 = x2.view(-1, 21, 64, 64, 64)
        joint_heatmap3d = torch.cat((x1, x2), 1)

        img_feat_gap = F.avg_pool2d(x, (x.shape[2], x.shape[3])).view(-1, 2048)
        # img_feat_gap.shape: torch.Size([16, 2048])
        root_heatmap1d = self.root_fc(img_feat_gap)
        # root_heatmap1d.shape: torch.Size([16, 64])
        root_depth = self.soft_argmax_1d(root_heatmap1d).view(-1, 1)
        # root_depth.shape: torch.Size([16, 1])
        hand_type = torch.sigmoid(self.hand_fc(img_feat_gap))
        # hand_type.shape: torch.Size([16, 2])
        return joint_heatmap3d, root_depth, hand_type

@register_model
def pvig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9  # 邻居节点数，默认为9
            self.conv = 'mr'  # 图卷积层类型，可选 {edge, mr}
            self.act = 'gelu'  # 激活层类型，可选 {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # 归一化方式，可选 {batch, instance}
            self.bias = True  # 卷积层是否使用偏置
            self.dropout = 0.0  # dropout率
            self.use_dilation = True  # 是否使用扩张knn
            self.epsilon = 0.2  # gcn的随机采样率
            self.use_stochastic = False  # gcn的随机性
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2, 1, 1, 1]  # 各层的block个数
            # self.channels = [80, 160, 400, 640]  # 各层的通道数
            self.channels = [256, 512, 1024, 2048, 1024, 512, 1344]   # 各层的通道数
            self.n_classes = num_classes  # 分类器输出通道数
            self.emb_dims = 1024  # 嵌入尺寸

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model

class GAT_PoseNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GAT_PoseNet, self).__init__()

        self.GATConv1 = GATConv(input_dim,1024)
        self.bn1 = BatchNorm(1024)

        self.GATConv2 = GATConv(1024,1024)
        self.bn2 = BatchNorm(1024)

        self.linear = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.GATConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.GATConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.linear(x)

        out = F.log_softmax(x, dim=1)

        return out






