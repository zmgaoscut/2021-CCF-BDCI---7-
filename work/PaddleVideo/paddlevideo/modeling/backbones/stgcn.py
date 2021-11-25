# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import weight_norm
import numpy as np
from ..registry import BACKBONES
from ..weight_init import weight_init_
import pdb

def zero(x):
    return 0


def iden(x):
    return x


def einsum(x, A):
    """paddle.einsum will be implemented in release/2.2.
    """
    x = x.transpose((0, 2, 3, 1, 4)) # nkctv->nctkv
    n, c, t, k, v = x.shape
    k2, v2, w = A.shape
    assert (k == k2 and v == v2), "Args of einsum not match!"
    x = x.reshape((n, c, t, k * v))
    A = A.reshape((k * v, w))
    y = paddle.matmul(x, A) #nctw=cntv
    return y 


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class Graph():
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout) #获取各个结点之间的连接关系
        self.hop_dis = get_hop_distance(self.num_node, # 获取结点间的距离信息，0,1或无穷（表示两个结点不相连）
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy) #得到sumj Dj**（-1/2）Aj Dj**（-1/2），记为A

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'fsd10':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                             (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                             (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                             (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                             (21, 14), (19, 14), (20, 19)]
            self.edge = self_link + neighbor_link
            self.center = 8
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy): #根据划分方案划分邻近矩阵,设计关节特征提取器A
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial': #只有spatial划分方案
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop: #两个结点的距离为0或1，为0表示同一个结点
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:  #同一结点才满足
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]: #结点j距离重心更近
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


class ConvTemporalGraphical(nn.Layer): # 图卷积
    # 图卷积的输出特征(n,c,t,v), 经过所有模块，v一直为25，t也没变，等于帧数，只改变了通道数c
    # 先用一个卷积将输入特征通道数扩大kernel_size（和A的k相等）倍，再和A相乘计算后，通道数变回c
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2D(in_channels,
                              out_channels * kernel_size, # 目的在于使得输出和A的维数匹配
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1))

    def forward(self, x, A):
        assert A.shape[0] == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.reshape((n, self.kernel_size, kc // self.kernel_size, t, v))
        x = einsum(x, A)

        return x, A


# reference
# "Squeeze-and-Excitation Networks",  <https://arxiv.org/abs/1709.01507>
class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape((b, c)) #对应Squeeze操作
        y = self.fc(y).reshape((b, c, 1, 1)) #对应Excitation操作
        return x * paddle.expand_as(y,x)

class st_gcn_block(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn_block, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        # 图卷积的输出特征(n,c,t,v), 经过所有模块，v一直为25，t也没变，等于帧数（例如350），只改变了通道数c
        # 先用一个卷积将输入特征通道数扩大kernel_size（和A的k相等）倍，再和A相乘计算后，通道数变回c
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        # 时域卷积由于有padding，所以t，v均没有改变,但padding取决于kernel size
        self.tcn = nn.Sequential(    
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2D(out_channels),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2D(out_channels),
            )

        self.relu = nn.ReLU()


        #########channel attention
        self.se_block = SELayer(channel=out_channels, reduction=16)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.se_block(self.tcn(x)) + res
        return self.relu(x), A


@BACKBONES.register()
class STGCN(nn.Layer):
    """
    ST-GCN model from:
    `"Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" <https://arxiv.org/abs/1801.07455>`_
    Args:
        in_channels: int, channels of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default 2.
        edge_importance_weighting: bool, whether to use edge attention. Default True.
        data_bn: bool, whether to use data BatchNorm. Default True.
    """
    def __init__(self,
                 in_channels=2,
                 edge_importance_weighting=True,
                 data_bn=True,
                 layout='fsd10',
                 strategy='spatial',
                 manifold_mixup = False,
                 **kwargs):
        super(STGCN, self).__init__()
        self.manifold_mixup = manifold_mixup
        self.data_bn = data_bn
        # load graph
        self.graph = Graph(
            layout=layout,
            strategy=strategy,
        )
        A = paddle.to_tensor(self.graph.A, dtype='float32') # A:(K,V,V),其中K表示划分的子集数（strategy = 'spatial'时，K=3），V为关节数
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.shape[0]  # 空间核大小为K
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # 1个batch中，一个结点归一化：该结点的所有帧、一个batch中的所有样本计算均值和方差
        # 2*结点数=2*25
        self.data_bn = nn.BatchNorm1D(in_channels *     
                                      A.shape[1]) if self.data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        ####### no dropout in the first layer,dropout's setting in backbone
        self.st_gcn_networks = nn.LayerList((
            st_gcn_block(in_channels,64,kernel_size, 1,residual=False,**kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        # 每个stgcn模块有一个权重矩阵M，大小为(K,V,V),即A的大小
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                self.create_parameter(
                    shape=self.A.shape,
                    default_initializer=nn.initializer.Constant(1))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def init_weights(self):
        """Initiate the parameters.
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                weight_init_(layer, 'Normal', mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2D):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm1D):
                weight_init_(layer, 'Normal', mean=1.0, std=0.02)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.shape
        x = x.transpose((0, 4, 3, 1, 2))  # N, M, V, C, T
        x = x.reshape((N * M, V * C, T))
        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)
        x = x.reshape((N, M, V, C, T))
        x = x.transpose((0, 1, 3, 4, 2))  # N, M, C, T, V
        x = x.reshape((N * M, C, T, V))

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, paddle.multiply(self.A, importance)) #A一直不变，通过加权来改变A作用于不同模块的作用

        if self.training and self.manifold_mixup:
            ############Manifold Mixup#####################
            # reference
            # "Manifold Mixup: Better Representations by Interpolating Hidden States"
            #  <https://arxiv.org/abs/1806.05236>
            index = paddle.randperm(N*M)
            lam = np.random.beta(0.2,0.2)
            x = x*lam + x[index]*(1-lam)


            x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
            C = x.shape[1]
            x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1
            return (x, lam, index)
        else:
            x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
            C = x.shape[1]
            x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1
            return x
