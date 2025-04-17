from typing import NamedTuple, List, Tuple

import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import dgl
from dgl import backend as F
from dgl.heterograph import DGLBlock


def gas_collate_fn(g, n_id, num_neighbors=-1):
    # import ipdb; ipdb.set_trace()

    n_id = n_id.to(g.device)
    frontier = dgl.sampling.sample_neighbors(g, n_id, num_neighbors,)
    seed_nodes = dgl.utils.prepare_tensor(g, n_id, 'items')
    block = dgl.to_block(frontier, seed_nodes)

    return block

class SubData(NamedTuple):
    data: DGLBlock
    batch_size: int
    n_id: Tensor  # The indices of mini-batched nodes
    offset: Tensor  # The offset of contiguous mini-batched nodes
    count: Tensor  # The number of contiguous mini-batched nodes

    def to(self, *args, **kwargs):
        return SubData(self.data.to(*args, **kwargs), self.batch_size,
                       self.n_id, self.offset, self.count)

class SubgraphLoader(DataLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors)."""
    def __init__(self, graph, merge_cluster, online_sampling, ptr: Tensor, deg_drop=-1, batch_size: int = 1,
                 bipartite: bool = True, log: bool = True, **kwargs):
        
        if merge_cluster:
            ptr = ptr[::batch_size]
            if int(ptr[-1]) != graph.num_nodes():
                ptr = torch.cat([ptr, torch.tensor([graph.num_nodes()], dtype=ptr.dtype)], dim=0)
            batch_size = 1

        self.graph = graph
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log
        self.online_sampling = online_sampling
        self.deg_drop = deg_drop

        n_id = torch.arange(graph.num_nodes(), dtype=ptr.dtype)
        batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
        batches = [(i, batches[i]) for i in range(len(batches))]

        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
        #     1, prefetch_node_feats=["h"]
        # )
        # import ipdb; ipdb.set_trace()
        # sampler.sample_blocks(self.graph, batches[0][1])

        if batch_size > 1 or self.online_sampling:
            super().__init__(batches, batch_size=batch_size,
                             collate_fn=self.compute_subgraph, **kwargs)

        else:  # If `batch_size=1`, we pre-process the subgraph generation:
            if log:
                t = time.perf_counter()
                print('Pre-processing subgraphs...', end=' ', flush=True)

            data_list = list(
                DataLoader(batches, collate_fn=self.compute_subgraph,
                           batch_size=batch_size, **kwargs))

            if log:
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

            super().__init__(data_list, batch_size=batch_size,
                             collate_fn=lambda x: x[0], **kwargs)


    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]) -> SubData :
        batch_ids, n_ids = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)

        # We collect the in-mini-batch size (`batch_size`), the offset of each
        # partition in the mini-batch (`offset`), and the number of nodes in
        # each partition (`count`)
        batch_size = n_id.numel()
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset)

        subgraph = gas_collate_fn(self.graph, n_id, self.deg_drop)

        return SubData(subgraph, batch_size, subgraph.srcdata[dgl.NID].cpu().long(), offset.long(), count.long())

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class EvalSubgraphLoader(SubgraphLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors).
    In contrast to :class:`SubgraphLoader`, this loader does not generate
    subgraphs from randomly sampled mini-batches, and should therefore only be
    used for evaluation.
    """
    def __init__(self, graph, merge_cluster, online_sampling, ptr: Tensor, deg_drop=-1, batch_size: int = 1,
                 bipartite: bool = True, log: bool = True, **kwargs):

        ptr = ptr[::batch_size]
        if int(ptr[-1]) != graph.num_nodes():
            ptr = torch.cat([ptr, torch.tensor([graph.num_nodes()], dtype=ptr.dtype)], dim=0)

        super().__init__(graph=graph, merge_cluster=True, online_sampling=online_sampling, ptr=ptr, deg_drop=-1, batch_size=1, bipartite=bipartite,
                         log=log, shuffle=False, num_workers=0, **kwargs)
