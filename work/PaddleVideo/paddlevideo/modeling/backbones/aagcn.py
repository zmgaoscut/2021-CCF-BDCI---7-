  
import math
import numpy as np
import paddle
import paddle.nn as nn
from ..weight_init import weight_init_
from ..registry import BACKBONES
import numpy as np


# reference
# "Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks"
# <https://arxiv.org/pdf/1912.06971v1.pdf>
# 
# @inproceedings{2sagcn2019cvpr,  
#       title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},  
#       author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
#       booktitle = {CVPR},  
#       year      = {2019},  
# }
# @article{shi_skeleton-based_2019,
#     title = {Skeleton-{Based} {Action} {Recognition} with {Multi}-{Stream} {Adaptive} {Graph} {Convolutional} {Networks}},
#     journal = {arXiv:1912.06971 [cs]},
#     author = {Shi, Lei and Zhang, Yifan and Cheng, Jian and LU, Hanqing},
#     month = dec,
#     year = {2019},
# }

### 用 Paddle 实现 ###


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward =[(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                            (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                            (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                            (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                            (21, 14), (19, 14), (20, 19)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.shape[0]
    k1 = weight.shape[1]
    k2 = weight.shape[2]
    weight_init_(conv,'Normal',mean=0.0, std=math.sqrt(2. / (n * k1 * k2 * branches)))


def conv_init(conv):
    weight_init_(conv, 'KaimingNormal')



def bn_init(bn, scale):
    weight_init_(bn, 'Constant', value=scale)


class unit_tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, adaptive=True, attention=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        self.conv_d = nn.LayerList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2D(in_channels, out_channels, 1))

        if adaptive:
            PA = self.create_parameter(shape=A.shape,default_initializer=nn.initializer.Assign(A.astype(np.float32)))
            self.add_parameter('PA',PA)
            alpha = self.create_parameter(shape=[1],default_initializer=nn.initializer.Assign(np.ones(1)))
            self.add_parameter('alpha',alpha) 
            self.conv_a = nn.LayerList()
            self.conv_b = nn.LayerList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2D(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2D(in_channels, inter_channels, 1))
        else:
            self.A = paddle.to_tensor(A.astype(np.float32))
        self.adaptive = adaptive

        if attention:

            # temporal attention
            self.conv_ta = nn.Conv1D(out_channels, 1, 9, padding=4)
            weight_init_(self.conv_ta,'Constant',value=0)


            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1D(out_channels, 1, ker_jpt, padding=pad)

            weight_init_(self.conv_sa,'XavierNormal')

            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            weight_init_(self.fc1c, 'KaimingNormal')

            weight_init_(self.fc2c, 'Constant',value=0)

        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2D):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.shape

        y = None
        if self.adaptive:
            A = self.PA

            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).transpose((0, 3, 1, 2)).reshape((N, V, self.inter_c * T))
                A2 = self.conv_b[i](x).reshape((N, self.inter_c * T, V))
                A1 = self.tan(paddle.matmul(A1, A2) / A1.shape[-1])  # N V V
                A1 = A[i] + A1 * self.alpha
                A2 = x.reshape((N, C * T, V))
                z = self.conv_d[i](paddle.matmul(A2, A1).reshape((N, C, T, V)))
                y = z + y if y is not None else z
        else:
            A = self.A * self.mask
            for i in range(self.num_subset):
                A1 = A[i]
                A2 = x.reshape((N, C * T, V))
                z = self.conv_d[i](paddle.matmul(A2, A1).reshape((N, C, T, V)))
                y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y


            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y


class TCN_GCN_unit(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        if self.attention:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

        else:
            y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y
@BACKBONES.register()
class AAGCN(nn.Layer):
    def __init__(self, num_class=30, num_point=25, num_person=1, in_channels=2,drop_out=0, adaptive=True, attention=True):
        super(AAGCN, self).__init__()

        self.graph = Graph()

        A = self.graph.A
        self.num_class = num_class

        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=attention)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=attention)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=attention)

        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
    def forward(self, x):
        N, C, T, V, M = x.shape

        x = x.transpose((0, 4, 3, 1, 2)).reshape((N, M * V * C, T))
        if self.data_bn:
            x.stop_gradient = False
        x = self.data_bn(x)
        x = x.reshape((N, M, V, C, T)).transpose((0, 1, 3, 4, 2)).reshape((N * M, C, T, V))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1
        x = self.drop_out(x)

        return x
