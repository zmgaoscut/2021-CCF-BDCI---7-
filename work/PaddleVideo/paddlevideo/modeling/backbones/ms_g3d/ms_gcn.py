
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from .tools import k_adjacency, normalize_adjacency_matrix
from .mlp import MLP
from .activation import activation_factory


class MultiScale_GraphConv(nn.Layer):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu'):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = paddle.to_tensor(A_powers)
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = self.create_parameter(
                    shape=self.A_powers.shape,
                    default_initializer=nn.initializer.Uniform(low=-1e-6, high=1e-6))
            # self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)


        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    def forward(self, x):
        N, C, T, V = x.shape
        # self.A_powers = self.A_powers.to(x.device)
        A = self.A_powers.astype(x.dtype)
        if self.use_mask:
            A = A + self.A_res.astype(x.dtype)

        support = einsum(x,A)
        support = support.reshape((N, C, T, self.num_scales, V))
        support = support.transpose((0,3,1,2,4)).reshape((N, self.num_scales*C, T, V))
        # support = torch.einsum('vu,nctu->nctv', A, x)
        # support = support.view(N, C, T, self.num_scales, V)
        # support = support.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(support)
        return out

def einsum(x, A):
    """paddle.einsum will be implemented in release/2.2.
    """
    n, c, t, u = x.shape
    v, u2 = A.shape
    assert (u==u2), "Args of einsum not match!"
    A = A.transpose((1,0))
    y = paddle.matmul(x, A) #nctv
    return y 

# if __name__ == "__main__":
#     from graph.ntu_rgb_d import AdjMatrixGraph
#     graph = AdjMatrixGraph()
#     A_binary = graph.A_binary
#     msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
#     msgcn.forward(torch.randn(16,3,30,25))
