import torch
import torch.nn as nn
from einops import rearrange

from .spatial_conv import SpatialConvOrderK, TemporalConvOrderK
from .temporal_conv import TemporalConv, LayerNorm
from .gcrnn import GCGRUCell
from .spatial_attention import SpatialAttention, FeatureAttention, TemporalAttention
from ..utils.ops import reverse_tensor
from .MIC import MIC
import random

import torch.nn.functional as F
import copy

class graph_constructor(nn.Module):
    def __init__(self, nnodes, dim, device, alpha=3, static_feat=None, order=3):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.order = order

        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, predefined_A):
        ones = torch.ones_like(predefined_A)
        adjacentA = torch.where(predefined_A > 0., ones, predefined_A)

        idx = torch.arange(self.nnodes).to(self.device)
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = torch.sigmoid(self.alpha*a)


        # InnerAdjacentA = adjacentA.to(self.device)
        #
        # adj01list = [InnerAdjacentA]
        # for i in range(self.order - 1):
        #     adj01i = torch.mm(adj01list[i], InnerAdjacentA)
        #     adj01i = torch.clamp(adj01i, max=1.0)
        #     diagi = torch.diag(torch.diag(adj01i))
        #     adj01i = adj01i - diagi
        #     for jadj in adj01list:
        #         adj01i = adj01i - jadj
        #     adj01i = torch.clamp(adj01i, min=0.0)
        #     adj01list.append(adj01i)

        # adjlist = []
        # for i in range(len(adj01list)):
        #     adjlist.append(adj * adj01list[i])

        adjlist = [adj * adjacentA]
        return adjlist

class SpatialDecoder(nn.Module):
    def __init__(self, d_in, d_model, d_out, support_len, order=1, attention_block=False, nheads=2, dropout=0., setdim = 3):
        super(SpatialDecoder, self).__init__()
        self.order = order
        if setdim == 3:
            self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        elif setdim == 4:
            self.lin_in = nn.Conv2d(d_in, d_model, kernel_size=1)
            self.lin_in2 = nn.Conv2d(d_in, d_model, kernel_size=1)
        self.graph_conv = SpatialConvOrderK(c_in=d_model, c_out=d_model,
                                            support_len=support_len * order, order=1, include_self=False)
        self.temp_graph_conv = TemporalConvOrderK(c_in=d_model, c_out=d_model,
                                            support_len=2 * order, order=1, include_self=False)

        if attention_block:
            self.spatial_att = SpatialAttention(d_in=d_model,
                                                d_model=d_model,
                                                nheads=nheads,
                                                dropout=dropout)
            if setdim == 3:
                self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
            elif setdim == 4:
                self.lin_out = nn.Conv2d(2 * d_model, d_model, kernel_size=1)
        else:
            self.register_parameter('spatial_att', None)
            if setdim == 3:
                self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
            elif setdim == 4:
                self.lin_out = nn.Conv2d(2 * d_model, d_model, kernel_size=1)

        if setdim == 3:
            self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        elif setdim == 4:
            self.read_out = nn.Conv2d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()
        self.adj = None
        self.context_attention = TemporalAttention(d_in=24,
                                                d_model=24,
                                                nheads=8,
                                                dropout=dropout)



        self.temporal_aggre = Multi_ScaleTCN_MeanAgg(cin=d_model, cout=d_model, dropout=dropout)


    def forward(self, x, m, h, u, spa_adj, tem_adj=None, cached_support=False):
        # spatial convolution
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, dim=1)
        if self.order > 1:
            if cached_support and (self.adj is not None):
                adj = self.adj
            else:
                adj = SpatialConvOrderK.compute_support_orderK(spa_adj, self.order, include_self=False, device=x_in.device)
                self.adj = adj if cached_support else None
        x_in = self.lin_in(x_in)

        out = self.graph_conv(x_in, spa_adj)

        if self.spatial_att is not None:
            # [batch, channels, nodes] -> [batch, steps, nodes, features]
            x_in = rearrange(x_in, 'b f n -> b 1 n f')
            out_att = self.spatial_att(x_in, torch.eye(x_in.size(2), dtype=torch.bool, device=x_in.device))
            out_att = rearrange(out_att, 'b s n f -> b f (n s)')
            out = torch.cat([out, out_att], 1)

        out = torch.cat([out, h], 1)
        out2 = self.activation(self.lin_out(out))
        # out = self.lin_out(out)
        out = torch.cat([out2, h], 1)
        return self.read_out(out), out, out2

class Multi_ScaleTCN(nn.Module):
    def __init__(self, cin, cout):
        super(Multi_ScaleTCN, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3]
        self.dilation_factor = [1, 2]
        cout = int(cout/(len(self.kernel_set) * len(self.dilation_factor)))
        for kern in self.kernel_set:
            for dila in self.dilation_factor:
                self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dila)))
    def forward(self,input):
        x = []
        zero_length = [1, 2, 2, 4] # corresponding dila = [1, 2]
        # zero_length = [2, 3, 4, 6]
        for i in range(len(self.kernel_set)* len(self.dilation_factor)):
            i_input = nn.functional.pad(input, (zero_length[i],0))
            x.append(self.tconv[i](i_input))
        x = torch.cat(x,dim=1)
        return x

class Multi_ScaleTCN_MeanAgg(nn.Module):
    def __init__(self, cin, cout, dropout):
        super(Multi_ScaleTCN_MeanAgg, self).__init__()
        self.tconv = nn.ModuleList()
        self.updownsampling = nn.ModuleList()
        self.downupsampling = nn.ModuleList()
        self.kernel_set = [2, 3]
        self.dilation_factor = [1, 2]
        for kern in self.kernel_set:
            for dila in self.dilation_factor:
                self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dila)))
        self.read_out = nn.Conv2d(4 * cout, cout, kernel_size=(1, 1))

        self.conv_out = nn.Conv2d(cout, cout, kernel_size=(1, 4))

        self.feature_attention = FeatureAttention(d_in=4 * cout,
                                                  d_model=cout,
                                                  nheads=2,
                                                  dropout=dropout)



    def forward(self,input):
        x = []
        zero_length = [1, 2, 2, 4] # corresponding dila = [1, 2]
        # zero_length = [2, 3, 4, 6]

        for i in range(len(self.kernel_set)* len(self.dilation_factor)):
            i_input = nn.functional.pad(input, (zero_length[i], 0))
            x.append(self.tconv[i](i_input))

        x1 = torch.cat(x,dim=1)
        x1 = self.read_out(x1)


        # x2 = torch.stack(x, -1)
        # x = torch.mean(x,dim=-1)
        return x1


class GRIL(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 u_size=None,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 support_len=2,
                 n_nodes=None,
                 layer_norm=False):
        super(GRIL, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        self.randTemGraph = torch.randn([24, 24])
        self.randSpaGraph = torch.randn([207, 207])
        rnn_input_size = 2 * self.input_size + self.u_size  # input + mask + (eventually) exogenous
        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.subseq_len = 4


        # Fist stage readout
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size , out_channels=self.input_size, kernel_size=1)

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.graph_conv = SpatialConvOrderK(c_in=hidden_size, c_out=hidden_size,
                                            support_len=2, order=1, include_self=False)

        self.local_field = 1
        self.spatial_decoder = SpatialDecoder(d_in=2 * self.input_size + self.local_field - 1 + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              order=decoder_order,
                                              attention_block=global_att, setdim=4)
        # self.temporal_aggre = Multi_ScaleTCN(cin=self.hidden_size,cout=self.hidden_size)
        self.temporal_aggre2 = Multi_ScaleTCN_MeanAgg(cin=self.hidden_size,cout=self.hidden_size, dropout=dropout)
        # self.temporal_aggre3 = MIC(feature_size=self.input_size,n_heads=8, dropout=0.05, decomp_kernel=[32],
        #                            conv_kernel=[3], isometric_kernel=[8])


        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0, self.h1 = self.init_hidden_states(n_nodes, self.subseq_len)
        else:
            self.register_parameter('h0', None)


        #  207 for METR-LA, 325 for PEMS_BAY, 437 for AIR, and 36 for AIR-36
        self.graph = graph_constructor(207, 64, 'cuda:0', order=1)

        # self.get_repr = nn.Conv2d(in_channels=1 + self.hidden_size, out_channels=self.hidden_size, kernel_size=1)
        self.win_len = 24


        self.fwd_tcn = nn.Conv2d(in_channels=1, out_channels=1,
                                 kernel_size=(1, 3))

        self.fwd_transtcn = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                               kernel_size=(1, 3))


    def temporalAgg(self, data):
        b, s, n, f = data.size()
        x = rearrange(data, 'b s n f -> b (s n) f')
        x = x.unsqueeze(dim=1)
        x = self.fwd_tcn(x)
        x = self.fwd_transtcn(x)
        x = x.squeeze(dim=1)
        x = rearrange(x, 'b (s n) f -> b s n f', b=b, s=s, n=n, f=f)
        return x

    def init_hidden_states(self, n_nodes, len):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            # h0.append(vals)
            h0.append(nn.Parameter(vals))

        # vals2 = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes, 24))
        vals2 = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes, len))
        h1 = nn.Parameter(vals2)
        # return h0, h1
        return nn.ParameterList(h0), h1


    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def get_h1(self, x):
        if self.h1 is not None:
            return self.h1.view(1, *self.h1.shape).expand(x.shape[0], -1, -1, -1)
        else:
            return torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2], x.shape[3])).to(x.device)

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h


    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        # x:[batch, features, nodes, steps]
        *_, steps = x.size()

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is not None:
            pass
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]


        # Temporal conv
        predictions, imputations, states = [], [], []

        adj2 = []
        adaptive_graph = self.graph(adj[0])[0]
        d = adaptive_graph.sum(1) + 1e-8
        adj2.append(adaptive_graph/d.view(-1, 1))
        trans_adaptive_graph = adaptive_graph.transpose(0,1)
        trans_d = trans_adaptive_graph.sum(1) + 1e-8
        adj2.append(trans_adaptive_graph/trans_d.view(-1, 1))

        pass

        # recurrent but efficient version
        imputed_x = []
        imputed_h = []
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            xs_hat_1 = self.first_stage(h_s)
            x_s_1 = torch.where(m_s, x_s, xs_hat_1)
            h = self.update_state(torch.cat([x_s_1, m_s], dim=1), h, adj)
            imputed_x.append(x_s_1)
            imputed_h.append(h_s)
            predictions.append(xs_hat_1)
        imputed_x = torch.stack(imputed_x, dim=-1)
        imputed_h = torch.stack(imputed_h, dim=-1)
        predictions = torch.stack(predictions, dim=-1)

        # feature fusion based on local temporal dynamics.
        imputed_h = self.temporal_aggre2(imputed_h)

        imputations, representations,_ = self.spatial_decoder(x=imputed_x, m=mask, h=imputed_h, u=u, spa_adj=adj2,
                                                            cached_support=cached_support)
        return imputations, predictions, representations, states


class BiGRIL(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 ff_dropout,
                 n_layers=1,
                 dropout=0.,
                 n_nodes=None,
                 support_len=2,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 u_size=0,
                 embedding_size=0,
                 layer_norm=False,
                 merge='mlp'):
        super(BiGRIL, self).__init__()
        self.fwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)
        self.bwd_rnn = GRIL(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            dropout=dropout,
                            n_nodes=n_nodes,
                            support_len=support_len,
                            kernel_size=kernel_size,
                            decoder_order=decoder_order,
                            global_att=global_att,
                            u_size=u_size,
                            layer_norm=layer_norm)

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)
        self.win_len = 24
        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + 1 + embedding_size,
                          out_channels=ff_size, kernel_size=1), # kernel_size=1
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=1, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None


    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
        # Forward
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        # # Backward
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        *bwd_res, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]

        # bwd_out, bwd_pred, bwd_repr = fwd_out, fwd_pred, fwd_repr
        if self._impute_from_states:
            # inputs = [fwd_repr, bwd_repr, mask]
            inputs = [fwd_repr, bwd_repr, mask]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1)


            imputation = self.out(imputation)

        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)

        return imputation, predictions