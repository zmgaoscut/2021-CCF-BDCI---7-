
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .ms_g3d.ms_gcn import MultiScale_GraphConv as MS_GCN
from .ms_g3d.ms_tcn import MultiScale_TemporalConv as MS_TCN
from .ms_g3d.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from .ms_g3d.mlp import MLP
from .ms_g3d.activation import activation_factory
from .ms_g3d import tools

from ..registry import BACKBONES
def get_edge(layout):
    # edge is a list of [child, parent] paris
    if layout == 'fsd10':
        neighbor_link = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                            (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                            (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                            (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                            (21, 14), (19, 14), (20, 19)]
        return neighbor_link
    else:
        raise ValueError("Do Not Exist This Layout.")
    

class AdjMatrixGraph:
    def __init__(self,num_node,neighbor):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)

class Identity(nn.Layer):
    def __init__(self):
        super(Identity,self).__init__()
        pass
    def forward(self,x):
        return x

# reference
# "Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition"
# <https://arxiv.org/pdf/2003.14111v2.pdf>
# 
# @inproceedings{liu2020disentangling,
#   title={Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition},
#   author={Liu, Ziyu and Zhang, Hongwen and Chen, Zhenghao and Wang, Zhiyong and Ouyang, Wanli},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={143--152},
#   year={2020}
# }

### 用 paddle 实现 ###
class MS_G3D(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 activation='relu',
                 dropout=0):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = Identity() 
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True,
                dropout=dropout
            )
        )

        self.out_conv = nn.Conv3D(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.reshape((N, self.embed_channels_out, -1, self.window_size, V))
        # x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(axis=3)
        # x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3,5],
                 window_stride=1,
                 window_dilations=[1,1],
                 dropout=0):

        super().__init__()
        self.gcn3d = nn.LayerList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation,
                dropout=dropout
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum

@BACKBONES.register()
class MSG3D(nn.Layer):
    """
    MSG3D model from Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition
    """
    def __init__(self,num_point=25,
                 num_person=1,
                 num_gcn_scales=8,
                 num_g3d_scales=8,
                 graph='fsd10',
                 in_channels=2,
                 dropout=0):
        super(MSG3D, self).__init__()

        # Graph = import_class(graph)
        Graph = AdjMatrixGraph(num_point,get_edge(layout=graph))
        A_binary = Graph.A_binary

        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(in_channels, c1, A_binary, num_g3d_scales, window_stride=1,dropout=dropout)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, in_channels, c1, A_binary, dropout=dropout,disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2,dropout=dropout)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, dropout=dropout,disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2,dropout=dropout)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A_binary, dropout=dropout,disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        # self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.shape
        x = x.transpose((0, 4, 3, 1, 2))  # N, M, V, C, T
        x = x.reshape((N * M, V * C, T))
        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)
        x = x.reshape((N * M, V, C, T)).transpose((0,2,3,1)) # (N*M,C,T,V)

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x))
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x))
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x))
        x = self.tcn3(x)

        x = self.pool(x) # Global Average Pooling (Spatial+Temporal)
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1
        return x


