import torch.nn as nn
import torch

from einops import rearrange


class SpatialAttention(nn.Module):
    def __init__(self, d_in, d_model, nheads, dropout=0.):
        super(SpatialAttention, self).__init__()
        self.lin_in = nn.Linear(d_in, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)

    def forward(self, x, att_mask=None, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        b, s, n, f = x.size()
        x = rearrange(x, 'b s n f -> n (b s) f')
        x = self.lin_in(x)
        x = self.self_attn(x, x, x, attn_mask=att_mask)[0]
        x = rearrange(x, 'n (b s) f -> b s n f', b=b, s=s)
        return x

class FeatureAttention(nn.Module):
    def __init__(self, d_in, d_model, nheads, dropout=0.):
        super(FeatureAttention, self).__init__()
        self.lin_in_Q = nn.Linear(d_in, d_model)
        self.lin_in_K = nn.Linear(d_in, d_model)
        self.lin_in_V = nn.Linear(d_in, d_model)
        self.lin_out = nn.Conv1d(d_model, 1, kernel_size=1)
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=0.)

    def forward(self, y, att_mask=None, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        y = y.permute(0, 3, 2, 1)
        b, f, n, s = y.size()
        y = rearrange(y, 'b f n s -> b (f n) s')
        Q_x = self.lin_in_Q(y)
        V_x = self.lin_in_V(y)
        K_x = self.lin_in_K(y)
        x = self.self_attn(Q_x, K_x, V_x, attn_mask=att_mask)[0]
        x = rearrange(x, 'b (f n) s -> b f n s', f = f, n = n)
        x = x.permute(0, 3, 2, 1)
        return x

class TemporalAttention(nn.Module):
    def __init__(self, d_in, d_model, nheads, dropout=0.):
        super(TemporalAttention, self).__init__()
        self.lin_in = nn.Linear(d_in, d_model)
        self.lin_out = nn.Conv2d(d_model, 1, kernel_size=1)
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)

    def forward(self, x, att_mask=None, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        b, s, n, f = x.size()
        x = rearrange(x, 'b s n f -> b (s n) f')
        x = self.lin_in(x)
        x = self.self_attn(x, x, x, attn_mask=att_mask)[0]
        x = rearrange(x, 'b (s n) f -> b s n f', b=b, s=s, n=n, f=f)
        return x