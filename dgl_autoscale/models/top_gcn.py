from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d

from dgl.heterograph import DGLBlock
from dgl.nn import GraphConv
from dgl_autoscale.models import ScalableGNN
from dgl_autoscale import History
from dgl_autoscale.models import TOPScalableGNN

import dgl.function as fn


class TOPGraphConv(GraphConv):
    def forward(self, graph, feat, adj_t_sketch_compensate = None, sketch_codebook=None, inb_id=None) -> Tensor:
        # edge_weight = graph.edata["weight"]
        with graph.local_scope():
            weight = self.weight
            # graph.edata["_edge_weight"] = edge_weight
            aggregate_fn = fn.u_mul_e("h", "weight", "m")
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat = torch.matmul(feat, weight)
                graph.srcdata["h"] = feat
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if adj_t_sketch_compensate is not None:
                    rst = rst + adj_t_sketch_compensate @ (sketch_codebook.T @ feat[inb_id])
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if adj_t_sketch_compensate is not None:
                    rst = rst + adj_t_sketch_compensate @ (sketch_codebook.T @ feat[inb_id])
                if weight is not None:
                    rst = torch.matmul(rst, weight)


            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class TOPGCN(TOPScalableGNN):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0, bn_name: str = 'BatchNorm1d',
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)

        self.num_nodes = num_nodes

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual
        self.linear = linear

        self.num_layers = num_layers

        # if self.in_channels == 1:
        #     self.emb_layer = torch.nn.Embedding(num_nodes, embedding_dim=hidden_channels)
        #     in_channels = hidden_channels

        self.lins = ModuleList()
        if linear:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0 and not linear:
                in_dim = in_channels
            if i == num_layers - 1 and not linear:
                out_dim = out_channels
            conv = TOPGraphConv(in_dim, out_dim, norm='none', allow_zero_in_degree=True,)
            # conv = GraphConv(in_dim, out_dim, norm='none', allow_zero_in_degree=True,)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = eval(bn_name)(hidden_channels)
            self.bns.append(bn)

        # self.histories = torch.nn.ModuleList([
        #     History(num_nodes, hidden_channels, f'cpu')
        #     for _ in range(num_layers-1)
        # ])

        # self.device = f"cuda"


    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def initial_feature(self, x):
        if self.linear:
            return self.lins[0](x).relu_()
        else:
            return x
    
    def forward(self, g, x, adj_t_sketch_compensate=None, sketch_codebook=None, inb_id_list=None) -> Tensor:
        # edge_weight = g.edata["weight"]
        # edge_weight = g.edata['weight'] if 'weight' in g.edata else None
        # x = data.x
        # adj_t = data.adj_t

        # adj_t_sketch_compensate = data.adj_t_sketch_compensate
        # sketch_codebook = data.sketch_codebook

        # if self.in_channels == 1:
        #     x = self.emb_layer(x)

        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (conv, bn, ) in enumerate(zip(self.convs[:-1], self.bns, )):
            # self.test_histories[0][g.ndata['_ID'].cpu()] == x
            # self.test_histories[0][g.dstdata['_ID'].cpu()] == x
            x, adj_t_sketch_compensate_layer, sketch_codebook_layer, inb_id_layer = self.get_compensation(g, x, adj_t_sketch_compensate, sketch_codebook, inb_id_list, i)
            h = conv(g, x, adj_t_sketch_compensate = adj_t_sketch_compensate_layer, sketch_codebook=sketch_codebook_layer, inb_id=inb_id_layer)
            if self.batch_norm:
                h = bn(h,)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = h.relu_()
            # x = self.push_and_pull(emb_hist, x, *args)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        
        x, adj_t_sketch_compensate_layer, sketch_codebook_layer, inb_id_layer = self.get_compensation(g, x, adj_t_sketch_compensate, sketch_codebook, inb_id_list, -1)
        h = self.convs[-1](g, x, adj_t_sketch_compensate = adj_t_sketch_compensate_layer, sketch_codebook=sketch_codebook_layer, inb_id=inb_id_layer)

        if not self.linear:
            return h

        if self.batch_norm:
            h = self.bns[-1](h,)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[1](h,)

    
    

    @torch.no_grad()
    def forward_layer(self, layer, g, x, state, batch_size):
        # edge_weight = g.edata['weight'] if 'weight' in g.edata else None
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.linear:
                x = self.lins[0](x).relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](g, x, ) # edge_weight=edge_weight,

        if layer < self.num_layers - 1 or self.linear:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()

        if layer == self.num_layers - 1 and self.linear:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        return h