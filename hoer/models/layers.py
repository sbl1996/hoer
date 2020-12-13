import math
from difflib import get_close_matches
from typing import Union, Tuple, Optional, Sequence, Mapping

from cerberus import Validator

from typing import Mapping, Union, Sequence
import numpy as np
# noinspection PyUnresolvedReferences
import mindspore.nn as nn
# noinspection PyUnresolvedReferences
import mindspore.ops as ops
# noinspection PyUnresolvedReferences
from mindspore.common.initializer import HeNormal, HeUniform, Uniform

DEFAULTS = {
    'bn': {
        'momentum': 0.9,
        'eps': 1e-5,
        'affine': True,
        'track_running_stats': True,
        'sync': False,
        'device_num_each_group': 1,
    },
    'gn': {
        'groups': None,
        'channels_per_group': 16,
        'eps': 1e-5,
        'affine': True,
    },
    'activation': 'relu',
    'compat': False,
    'leaky_relu': {
        'alpha': 0.1,
    },
    'norm': 'bn',
    'init': {
        'type': 'msra',
        'mode': 'fan_in',
        'distribution': 'uniform',
        'fix': True,
    },
    'no_bias_decay': False,
}

_defaults_schema = {
    'bn': {
        'momentum': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'track_running_stats': {'type': 'boolean'},
        'fused': {'type': 'boolean'},
        'sync': {'type': 'boolean'},
        'device_num_each_group': {'type': 'integer'},
    },
    'gn': {
        'eps': {'type': 'float', 'min': 0.0},
        'affine': {'type': 'boolean'},
        'groups': {'type': 'integer'},
        'channels_per_group': {'type': 'integer'},
    },
    'activation': {'type': 'string', 'allowed': ['relu', 'swish', 'mish', 'leaky_relu', 'sigmoid']},
    'leaky_relu': {
        'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0},
    },
    'norm': {'type': 'string', 'allowed': ['bn', 'gn', 'none']},
    'init': {
        'type': {'type': 'string', 'allowed': ['msra', 'normal']},
        'mode': {'type': 'string', 'allowed': ['fan_in', 'fan_out']},
        'distribution': {'type': 'string', 'allowed': ['uniform', 'truncated_normal','untruncated_normal']},
        'fix': {'type': 'boolean'},
    },
    'no_bias_decay': {'type': 'boolean'},
}


def set_defaults(kvs: Mapping):
    def _set_defaults(kvs, prefix):
        for k, v in kvs.items():
            if isinstance(v, dict):
                _set_defaults(v, prefix + (k,))
            else:
                set_default(prefix + (k,), v)

    return _set_defaults(kvs, ())


def set_default(keys: Union[str, Sequence[str]], value):
    def loop(d, keys, schema):
        k = keys[0]
        if k not in d:
            match = get_close_matches(k, d.keys())
            if match:
                raise KeyError("No such key `%s`, maybe you mean `%s`" % (k, match[0]))
            else:
                raise KeyError("No key `%s` in %s" % (k, d))
        if len(keys) == 1:
            v = Validator({k: schema[k]})
            if not v.validate({k: value}):
                raise ValueError(v.errors)
            d[k] = value
        else:
            loop(d[k], keys[1:], schema[k])

    if isinstance(keys, str):
        keys = [keys]
    loop(DEFAULTS, keys, _defaults_schema)


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if padding == 'same':
        padding = [(k - 1) // 2 for k in kernel_size]
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init='he_uniform', has_bias=False, pad_mode="pad")


def calc_same_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation
    ph = (kh + (kh - 1) * (dh - 1) - 1) // 2
    pw = (kw + (kw - 1) * (dw - 1) - 1) // 2
    padding = (ph, pw)
    return padding


def Conv2d(in_channels: int,
           out_channels: int,
           kernel_size: Union[int, Tuple[int, int]],
           stride: Union[int, Tuple[int, int]] = 1,
           padding: Union[str, int, Tuple[int, int]] = 'same',
           groups: int = 1,
           dilation: int = 1,
           bias: Optional[bool] = None,
           norm: Optional[str] = None,
           act: Optional[str] = None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(padding, str):
        assert padding == 'same'
    if padding == 'same':
        padding = calc_same_padding(kernel_size, dilation)

    # Init
    init_cfg = DEFAULTS['init']
    if init_cfg['type'] == 'msra':
        mode = init_cfg['mode']
        distribution = init_cfg['distribution']
        if 'uniform' in distribution:
            weight_init = HeUniform(mode=mode)
        else:
            weight_init = HeNormal(mode=mode)
    else:
        raise ValueError("Unsupported init type: %s" % init_cfg['type'])

    scale = math.sqrt(1 / (kernel_size[0] * kernel_size[1] * (in_channels // groups)))
    bias_init = Uniform(scale)

    if bias is None:
        use_bias = norm is None
    else:
        use_bias = bias

    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, strides=stride,
                  pad_mode='pad', dilation=dilation, has_bias=use_bias, group=groups,
                  weight_init=weight_init, bias_init=bias_init)

    layers = [conv]

    if norm:
        layers.append(Norm(out_channels, norm))
    if act:
        layers.append(Act(act))

    if len(layers) == 1:
        return layers[0]
    else:
        return nn.SequentialCell(layers)


def Norm(channels, type='default', affine=None, track_running_stats=None, zero_init=False):
    if type in ['default', 'def']:
        type = 'bn'
    if type == 'bn':
        cfg = DEFAULTS['bn']
        if affine is None:
            affine = cfg['affine']
        if track_running_stats is None:
            track_running_stats = cfg['track_running_stats']
        if track_running_stats:
            use_batch_statistics = None
        else:
            use_batch_statistics = True
        if zero_init:
            gamma_init = 'zeros' if zero_init else 'ones'
        if cfg['sync']:
            bn = nn.GlobalBatchNorm(
                num_features=channels, momentum=cfg['momentum'], eps=cfg['eps'], affine=affine, gamma_init=gamma_init,
                use_batch_statistics=use_batch_statistics, device_num_each_group=cfg['device_num_each_group'])
        else:
            bn = nn.BatchNorm2d(
                num_features=channels, momentum=cfg['momentum'], eps=cfg['eps'], affine=affine, gamma_init=gamma_init,
                use_batch_statistics=use_batch_statistics)
        return bn
    elif type == 'none':
        return nn.Identity()
    else:
        raise ValueError("Unsupported normalization type: %s" % type)


def Act(type='default'):
    if type in ['default', 'def']:
        return Act(DEFAULTS['activation'])
    if type == 'relu':
        return ops.ReLU()
    elif type == 'sigmoid':
        return ops.Sigmoid()
    elif type == 'hswish':
        return nn.HSwish()
    elif type == 'leaky_relu':
        return nn.LeakyReLU(alpha=DEFAULTS['leaky_relu']['alpha'])
    else:
        raise ValueError("Unsupported activation type: %s" % type)


def Pool2d(kernel_size, stride, padding='same', type='avg', ceil_mode=False):
    assert padding == 0 or padding == 'same'
    if padding == 0:
        padding = 'valid'

    if type == 'avg':
        pool = nn.AvgPool2d
    elif type == 'max':
        pool = nn.MaxPool2d
    else:
        raise ValueError("Unsupported pool type: %s" % type)

    return pool(kernel_size, stride, padding)


def Linear(in_channels, out_channels, act=None):
    weight_init = HeUniform(mode='fan_in')
    scale = math.sqrt(1 / in_channels)
    bias_init = Uniform(scale)
    return nn.Dense(
        in_channels, out_channels, activation=act,
        weight_init=weight_init, bias_init=bias_init)


class GlobalAvgPool(nn.Cell):

    def __init__(self, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim

    def construct(self, x):
        return ops.reduce_mean(x, (2, 3), keep_dims=self.keep_dims)


class Identity(nn.Cell):

    def __init__(self):
        super().__init__()

    def construct(self, x):
        return x
