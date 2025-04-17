from typing import Optional, Callable, Dict, Any

import warnings

import torch
from torch import Tensor

from dgl_autoscale import History, AsyncIOPool
from dgl_autoscale import SubgraphLoader, EvalSubgraphLoader
from dgl.heterograph import DGLBlock

class TOPScalableGNN(torch.nn.Module):
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size = buffer_size

        self.histories = torch.nn.ModuleList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers - 1)
        ])

        self.pool: Optional[AsyncIOPool] = None
        self._async = False
        self.__out: Optional[Tensor] = None

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.histories[0]._device

    def _apply(self, fn: Callable) -> None:
        super()._apply(fn)
        # We only initialize the AsyncIOPool in case histories are on CPU:
        if (str(self.emb_device) == 'cpu' and str(self.device)[:4] == 'cuda'
                and self.pool_size is not None
                and self.buffer_size is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories[0].embedding_dim)
            self.pool.to(self.device)
        return self

    def reset_parameters(self):
        for history in self.histories:
            history.reset_parameters()
    
    
    def initial_feature(self, x):
        return x

    def get_compensation(self, graph, feat, adj_t_sketch_compensate, sketch_codebook, inb_id_list, layer):
        if isinstance(adj_t_sketch_compensate, list):
            adj_t_sketch_compensate_layer = adj_t_sketch_compensate[layer]
            sketch_codebook_layer = sketch_codebook[layer]
            inb_id = inb_id_list[layer]
        elif isinstance(adj_t_sketch_compensate, torch.nn.Sequential):
            if graph.num_src_nodes() != graph.num_dst_nodes():
                if layer == 0:
                    return feat, None, None, None
                else:
                    inb_id = inb_id_list
                    feat_ob = adj_t_sketch_compensate(feat[inb_id].T)
                    feat = torch.cat([feat[inb_id], feat_ob.T])
                    return feat, None, None, None
        else:
            adj_t_sketch_compensate_layer = adj_t_sketch_compensate
            sketch_codebook_layer = sketch_codebook
            inb_id = inb_id_list
        
        if adj_t_sketch_compensate_layer is None or len(adj_t_sketch_compensate_layer) == 0:
            return feat, None, None, None

        if graph.num_src_nodes() != graph.num_dst_nodes():
            if layer != 0:
                feat = torch.cat([feat, adj_t_sketch_compensate_layer @ (sketch_codebook_layer.T @ feat[inb_id])])
            return feat, None, None, None
        else:
            return feat, adj_t_sketch_compensate_layer, sketch_codebook_layer, inb_id

    @property
    def _out(self):
        if self.__out is None:
            self.__out = torch.empty(self.num_nodes, self.out_channels,
                                     pin_memory=True)
        return self.__out

    @torch.no_grad()
    def mini_inference_serial(self, loader: SubgraphLoader) -> Tensor:
        if self.pool is not None:
            return self.mini_inference(loader)
        states = [{} for i in range(len(loader))]

        # We push the outputs of the first layer to the history:
        for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
            subgraph = subgraph.to(self.device)
            x = subgraph.srcdata['feat']
            out = self.forward_layer(0, subgraph, x, state, batch_size)[:batch_size]
            self.histories[0].push(out, n_id[:batch_size], offset, count)

        for i in range(1, len(self.histories)):
            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
                subgraph = subgraph.to(self.device)
                x = self.histories[i - 1].pull(n_id.to(self.emb_device))
                out = self.forward_layer(i, subgraph, x, state, batch_size)[:batch_size]
                self.histories[i].push(out, n_id[:batch_size], offset, count)

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        out_all = torch.zeros(self.num_nodes, self.out_channels, device=self.device)
        for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
            subgraph = subgraph.to(self.device)
            x = self.histories[-1].pull(n_id.to(self.emb_device))
            out = self.forward_layer(self.num_layers - 1, subgraph, x,
                                     state, batch_size)[:batch_size]
            out_all[n_id[:batch_size]] = out

        return out_all.cpu()

    @torch.no_grad()
    def mini_inference(self, loader: SubgraphLoader) -> Tensor:
        r"""An implementation of layer-wise evaluation of GNNs.
        For each individual layer and mini-batch, :meth:`forward_layer` takes
        care of computing the next state of node embeddings.
        Additional state (such as residual connections) can be stored in
        a `state` directory."""

        # We iterate over the loader in a layer-wise fashsion.
        # In order to re-use some intermediate representations, we maintain a
        # `state` dictionary for each individual mini-batch.

        states = [{} for i in range(len(loader))]

        if self.pool is None:
            return self.mini_inference_serial(loader)

        # We push the outputs of the first layer to the history:
        for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
            subgraph = subgraph.to(self.device)
            x = subgraph.srcdata['feat']
            out = self.forward_layer(0, subgraph, x, state, batch_size)[:batch_size]
            self.pool.async_push(out, offset, count, self.histories[0].emb)
        self.pool.synchronize_push()

        for i in range(1, len(self.histories)):
            # Pull the complete layer-wise history:
            for _, batch_size, n_id, offset, count in loader:
                self.pool.async_pull(self.histories[i - 1].emb, offset, count,
                                     n_id[batch_size:])

            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
                subgraph = subgraph.to(self.device)
                x = self.pool.synchronize_pull()[:n_id.numel()]
                out = self.forward_layer(i, subgraph, x, state, batch_size)[:batch_size]
                self.pool.async_push(out, offset, count, self.histories[i].emb)
                self.pool.free_pull()
            self.pool.synchronize_push()

        # We pull the histories from the last layer:
        for _, batch_size, n_id, offset, count in loader:
            self.pool.async_pull(self.histories[-1].emb, offset, count,
                                 n_id[batch_size:])

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
            subgraph = subgraph.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            out = self.forward_layer(self.num_layers - 1, subgraph, x,
                                     state, batch_size)[:batch_size]
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool.synchronize_push()

        return self._out
    
    @torch.no_grad()
    def mini_inference_gpu(self, loader: SubgraphLoader) -> Tensor:
        states = [{} for i in range(len(loader))]

        histories = []

        y = torch.empty(
                self.num_nodes,
                self.hidden_channels,
                dtype=loader.graph.ndata['feat'].dtype,
                device=loader.graph.device,
                pin_memory=False,
        )
        # We push the outputs of the first layer to the history:
        for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
            subgraph = subgraph.to(self.device)
            x = subgraph.srcdata['feat']
            out = self.forward_layer(0, subgraph, x, state, batch_size)[:batch_size]
            y[offset : offset+count] = out.to(y.device)
        histories.append(y)

        for i in range(1, len(self.histories)):
            y = torch.empty(
                    self.num_nodes,
                    self.hidden_channels,
                    dtype=loader.graph.ndata['feat'].dtype,
                    device=loader.graph.device,
                    pin_memory=False,
            )
            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for (subgraph, batch_size, n_id, offset, count), state in zip(loader, states):
                subgraph = subgraph.to(self.device)
                x = histories[-1][n_id]
                out = self.forward_layer(i, subgraph, x, state, batch_size)[:batch_size]
                y[offset : offset+count] = out.to(y.device)
            histories.append(y)


        return histories

    @torch.no_grad()
    def forward_layer(self, layer: int, g: DGLBlock, x: Tensor,
                      state: Dict[str, Any]) -> Tensor:
        raise NotImplementedError
