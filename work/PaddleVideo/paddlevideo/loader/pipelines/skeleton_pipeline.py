#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import random
from ..registry import PIPELINES
import copy as cp
"""pipeline ops for Activity Net.
"""


@PIPELINES.register()
class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size: int, temporal size of skeleton feature.
        random_pad: bool, whether do random padding when frame length < window size. Default: False.
    """
    def __init__(self, window_size, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T

    def __call__(self, results):
        data = results['data']

        C, T, V, M = data.shape
        T = self.get_frame_num(data)
        if T == self.window_size:
            data_pad = data[:, :self.window_size, :, :]
        elif T < self.window_size:
            begin = random.randint(0, self.window_size -
                                   T) if self.random_pad else 0
            data_pad = np.zeros((C, self.window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        else:
            if self.random_pad:
                index = np.random.choice(T, self.window_size,
                                         replace=False).astype('int64')
            else:
                index = np.linspace(0, T-1, self.window_size).astype("int64")
            data_pad = data[:, index, :, :]

        results['data'] = data_pad
        return results


@PIPELINES.register()
class SkeletonNorm(object):
    """
    Normalize skeleton feature.
    Args:
        aixs: dimensions of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default: 2.
    """
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data[:2,:,:,:] = data[:2,:,:,:] - data[:2, :, 8:9, :]
        data = data[:self.axis, :, :, :]  # get (x,y) from (x,y, acc)
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1

        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class Iden(object):
    """
    Wrapper Pipeline
    """
    def __init__(self, label_expand=True):
        self.label_expand = label_expand

    def __call__(self, results):
        data = results['data']
        results['data'] = data.astype('float32')

        if 'label' in results and self.label_expand:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


import random



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

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):

    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

@PIPELINES.register()
class RandomMove(object):
    """
    Random move
    """
    def __init__(self):
        pass
    def __call__(self, results, 
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
        data = results['data']
        data_numpy = data    
        # input: C,T,V,M
        C, T, V, M = data_numpy.shape
        move_time = random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                        [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]  # pingyi bianhuan
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
        results['data'] = data_numpy

        return results

@PIPELINES.register()
class RandomShift(object):
    """
    Random shift
    """
    def __init__(self):
        pass
    def __call__(self,results):
        data = results['data']
        data_numpy = data 

        C, T, V, M = data_numpy.shape
        data_shift = np.zeros(data_numpy.shape)
        valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
        begin = valid_frame.argmax()
        end = len(valid_frame) - valid_frame[::-1].argmax()

        size = end - begin
        bias = random.randint(0, T - size)
        data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]
        results['data'] = data_numpy
        return results

@PIPELINES.register()
class OpenposeMatch(object):
    """
    OpenposeMatch
    """
    def __init__(self):
        pass
    def __call__(self,results):
        data = results['data']
        data_numpy = data
        C, T, V, M = data_numpy.shape
        assert (C == 3)
        score = data_numpy[2, :, :, :].sum(axis=1)
        # the rank of body confidence in each frame (shape: T-1, M)
        rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

        # data of frame 1
        xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
        # data of frame 2
        xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
        # square of distance between frame 1&2 (shape: T-1, M, M)
        distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

        # match pose
        forward_map = np.zeros((T, M), dtype=int) - 1
        forward_map[0] = range(M)
        for m in range(M):
            choose = (rank == m)
            forward = distance[choose].argmin(axis=1)
            for t in range(T - 1):
                distance[t, :, forward[t]] = np.inf
            forward_map[1:][choose] = forward
        assert (np.all(forward_map >= 0))

        # string data
        for t in range(T - 1):
            forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

        # generate data
        new_data_numpy = np.zeros(data_numpy.shape)
        for t in range(T):
            new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                                t]].transpose(1, 2, 0)
        data_numpy = new_data_numpy

        # score sort
        trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
        rank = (-trace_score).argsort()
        data_numpy = data_numpy[:, :, :, rank]
        results['data'] = data_numpy

        return results




