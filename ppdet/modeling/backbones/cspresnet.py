# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant

from ppdet.modeling.ops import get_act_fn
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import numpy
import cv2
import warnings

__all__ = ['CSPResNet', 'BasicBlock', 'EffectiveSELayer', 'ConvBNLayer']



#!!!!!!!!!!!!!!!have changed!!!!!!!!!!!
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 dilation=1,
                 weight_attr=None,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            weight_attr=weight_attr,
            bias_attr=False)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class RepVggBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, act='relu', alpha=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act
        if alpha:
            self.alpha = self.create_parameter(
                shape=[1],
                attr=ParamAttr(initializer=Constant(value=1.)),
                dtype="float32")
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.set_value(kernel)
        self.conv.bias.set_value(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu',
                 shortcut=True,
                 use_alpha=False):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return paddle.add(x, y)
        else:
            return y


class EffectiveSELayer(nn.Layer):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class DWConv(nn.Layer):
    """Depthwise Conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_size,
                 dilation=1,
                 stride=1,
                 bias=False,
                 padding=1,
                 act="silu"):
        super(DWConv, self).__init__()
        self.dw_conv = ConvBNLayer(
            in_channels,
            in_channels,
            filter_size=filter_size,
            stride=stride,
            groups=in_channels,
            dilation=dilation,
            padding=padding,
            act=act)
        self.pw_conv = ConvBNLayer(
            in_channels,
            out_channels,
            filter_size=1,
            stride=1,
            groups=1,
            act=act)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class ESPNet(nn.Layer):# paper
    def __init__(self, c, out_channels, act='silu',shortcut=True,use_alpha=False):
        super(ESPNet, self).__init__()
        self.conv11 = ConvBNLayer(
            ch_in = c, 
            ch_out = c // 4,
            filter_size=1,
            stride=1,
            padding=0,
            act=act)
        self.d1 = DWConv(
                in_channels = c // 4, 
                out_channels = c // 4,
                filter_size=3,
                dilation=1,
                stride=1,
                padding=1,act=act)       
       
        self.d2 = DWConv(
                in_channels = c // 4, 
                out_channels = c // 4,
                filter_size=3,
                dilation=2,
                stride=1,
                padding=2,act=act)
       
        self.d3 = DWConv(
                in_channels = c // 4, 
                out_channels = c // 4,
                filter_size=3,
                dilation=3,
                stride=1,
                padding=3,act=act)
        
        self.d4 = DWConv(
                in_channels = c // 4, 
                out_channels = c // 4,
                filter_size=3,
                dilation=4,
                stride=1,
                padding=4,act=act)
        #self.conv3 = nn.AvgPool2D(kernel_size=3, stride=1,padding=1)    
        
    def forward(self, x):
        out = []
        p = self.conv11(x)
        mid_layer1 = self.d1(p)
        out.append(mid_layer1)
        
        mid_layer2 = self.d2(p)
        mid_layer2 = paddle.add(mid_layer1,mid_layer2)
        out.append(mid_layer2)
        
        mid_layer3 = self.d3(p)
        mid_layer3 = paddle.add(mid_layer2,mid_layer3)
        out.append(mid_layer3)
        
        mid_layer4 = self.d4(p)
        mid_layer4 = paddle.add(mid_layer4,mid_layer3)
        out.append(mid_layer4)        
        
        y = paddle.concat(out,axis=1)
        y = paddle.add(x,y)
            
        #return self.conv3(y)
        return y


class CSPResStage(nn.Layer):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn='eca',
                 use_alpha=False,
                 end=False):
        super(CSPResStage, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2,
                ch_mid // 2,
                act=act,
                shortcut=True,
                use_alpha=use_alpha) for i in range(n)
        ])
        self.weight = self.create_parameter(
                shape=[2],
                attr=ParamAttr(initializer=Constant(value=1.)),
                dtype="float32")
        #self.weight = self.create_parameter(paddle.ones(1),dtype="float32")

        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
            #self.attn = Att(channels=ch_mid)
            #self.attn = TripletAttention()
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        w1 = paddle.exp(self.weight[0])/paddle.sum(paddle.exp(self.weight))
        w2 = paddle.exp(self.weight[1])/paddle.sum(paddle.exp(self.weight))
        y1 = self.conv1(x)
        y1 = w1*y1
        y2 = self.blocks(self.conv2(x))
        y2 = w2*y2
        y = paddle.concat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        #self.draw_CAM(y)
        return y


@register
@serializable
class CSPResNet(nn.Layer):
    __shared__ = ['width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[1, 2, 3],
                 depth_wise=False,
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 trt=False,
                 use_checkpoint=False,
                 use_alpha=False,
                 **args):
        super(CSPResNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act

        if use_large_stem:
            self.stem = nn.Sequential(
                ('conv1', ConvBNLayer(
                    3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                ('conv2', ConvBNLayer(
                    channels[0] // 2,
                    channels[0] // 2,
                    3,
                    stride=1,
                    padding=1,
                    act=act)), ('conv3', ConvBNLayer(
                        channels[0] // 2,
                        channels[0],
                        3,
                        stride=1,
                        padding=1,
                        act=act)))
        else:
            self.stem = nn.Sequential(
                ('conv1', ConvBNLayer(
                    3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                ('conv2', ConvBNLayer(
                    channels[0] // 2,
                    channels[0],
                    3,
                    stride=1,
                    padding=1,
                    act=act)))

        n = len(channels) - 1
        self.stages = nn.Sequential(*[(str(i), CSPResStage(
            BasicBlock  if i!=3 else ESPNet,
            channels[i],
            channels[i + 1],
            layers[i],
            2,
            act=act,
            use_alpha=use_alpha)) for i in range(n)])

        self._out_channels = channels[1:]
        self._out_strides = [4 * 2**i for i in range(n)]
        self.return_idx = return_idx
        if use_checkpoint:
            paddle.seed(0)

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            if self.use_checkpoint and self.training:
                x = paddle.distributed.fleet.utils.recompute(
                    stage, x, **{"preserve_rng_state": True})
            else:
                x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
   #    plt.subplot(1,1,1)
    #   print("len(outs):",len(outs))
   #    print(outs[0].detach().cpu()[0,0:1,:,:].numpy().transpose(1,2,0).shape)
    #   plt.imshow(outs[-1].detach().cpu()[0,0:1,:,:].numpy().transpose(1,2,0))
     #  plt.savefig("/home/yaozhuohan/bxy/PaddleDetection_YOLOSeries/ppdet/modeling/test1.jpg")
        return outs

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]
