import torch
from torch import nn
import numbers
from torch.nn import init
import torch.nn.functional as F

from ... import epsilon


class TemporalConv(nn.Module):
    """
    Spatial dilated convolution

    Efficient implementation inspired from graph-wavenet codebase
    """

    def __init__(self, c_in, c_out, win_len, dilation_factor=2, include_self=True):
        super(TemporalConv, self).__init__()
        self.include_self = include_self
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.win_len = win_len
        cout = int(c_out / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(c_in, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            i_repr = self.tconv[i](input)
            if i_repr.shape[-1] < self.win_len:
                i_repr = nn.functional.pad(i_repr, (self.win_len - i_repr.shape[-1], 0, 0, 0))
            x.append(i_repr)
        x = torch.cat(x, dim=1)
        return x

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)