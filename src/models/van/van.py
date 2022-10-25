#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : VAN model
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import mindspore.common.initializer as weight_init

from functools import partial

from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.common import Parameter, Tensor
from mindspore.ops import ReduceMean, Reshape, Transpose

from src.args import args
from src.models.van.misc import _ntuple, get_batch_norm_op, Identity, DropPath2D

to_2tuple = _ntuple(2)
BatchNorm2d = get_batch_norm_op(args)
print("====== BatchNorm OP: {} ======".format(BatchNorm2d), flush=True)


class DWConv(nn.Cell):
    """DWConv"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, pad_mode="pad", padding=1, has_bias=True, group=dim)

    def construct(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Cell):
    """MLP."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(
            in_features, hidden_features, kernel_size=1, stride=1, pad_mode='valid', has_bias=True)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1, pad_mode='valid', has_bias=True)
        self.drop = nn.Dropout(keep_prob=1-drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Cell):
    """Large Kernel Attentio."""

    def __init__(self, dim):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(
            dim, dim, kernel_size=5, stride=1, pad_mode="pad", padding=2, has_bias=True, group=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, kernel_size=7, stride=1, pad_mode="pad", padding=9, has_bias=True, group=dim, dilation=3)
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=1, stride=1, pad_mode='valid', has_bias=True)

    def construct(self, x):
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        out = u * attn
        return out


class Attention(nn.Cell):
    """Attention."""

    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.proj_1 = nn.Conv2d(
            d_model, d_model, kernel_size=1, pad_mode='valid', has_bias=True)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(
            d_model, d_model, kernel_size=1, pad_mode='valid', has_bias=True)

    def construct(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class Block(nn.Cell):
    """Block."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super(Block, self).__init__()
        self.dim = dim

        self.norm1 = BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath2D(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = Parameter(
            Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32), requires_grad=True)
        self.layer_scale_2 = Parameter(
            Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32), requires_grad=True)

    def construct(self, x):
        u0 = x
        x = self.norm1(x)
        x = self.attn(x)
        x = Reshape()(self.layer_scale_1, (self.dim, 1, 1)) * x
        x = self.drop_path(x)
        x = u0 + x

        u1 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = Reshape()(self.layer_scale_2, (self.dim, 1, 1)) * x
        x = self.drop_path(x)
        x = u1 + x

        return x


class OverlapPatchEmbed(nn.Cell):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        patch_size = to_2tuple(patch_size)
        padding = (patch_size[0] // 2, patch_size[0] // 2, patch_size[1] // 2, patch_size[1] // 2)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            pad_mode="pad", padding=padding, has_bias=True)
        self.norm = BatchNorm2d(embed_dim)

    def construct(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Cell):
    """Visual Attention Network."""

    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super(VAN, self).__init__()
        if not flag:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        cell_list = []
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])
            block = nn.CellList(
                [Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j]) for j in range(depths[i])])
            norm = norm_layer((embed_dims[i],))
            cur += depths[i]

            cell_list.append(patch_embed)
            cell_list.append(block)
            cell_list.append(norm)

        self.cell_list = nn.CellList(cell_list)

        # classification head
        self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()

        # operations
        self.reshape = Reshape()
        self.transpose = Transpose()
        self.mean = ReduceMean(keep_dims=False)
        self.init_weights()

    def init_weights(self,):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    weight_init.initializer(weight_init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(
                    weight_init.initializer(weight_init.Zero(), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(
                    weight_init.initializer(
                        weight_init.Normal(sigma=math.sqrt(2.0 / fan_out)), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
    
    def construct_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = self.cell_list[i * 3 + 0]
            block = self.cell_list[i * 3 + 1]
            norm = self.cell_list[i * 3 + 2]
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)

            s0 = x.shape[0]
            s1 = x.shape[1]
            x = self.reshape(x, (s0, s1, -1))
            x = self.transpose(x, (0, 2, 1))
            x = norm(x)

            if i != self.num_stages - 1:
                x = self.reshape(x, (B, H, W, -1))
                x = self.transpose(x, (0, 3, 1, 2))

        x = self.mean(x, 1)
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


def van_tiny(**kwargs):
    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    return model


def van_small(**kwargs):
    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    return model


def van_base(**kwargs):
    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    return model


def van_large(**kwargs):
    model = VAN(
        img_size=224, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    return model


def demo():
    data = np.random.rand(8, 3, 224, 224).astype(np.float32)

    m_input = Tensor(data, dtype=mstype.float32)

    van = van_tiny()

    out = van(m_input)

    print(out.shape, flush=True)

    import mindspore as ms
    ms.ops.Mul


if __name__ == "__main__":
    demo()
