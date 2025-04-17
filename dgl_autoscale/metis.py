from typing import List

import time
import torch
from torch import Tensor

import dgl
from dgl import backend as F

def sequence(graph, num_parts, recursive = False, log = True):
    if log:
        t = time.perf_counter()
        print(f'Computing randomly partitioning with {num_parts} parts...',)

    num_nodes = graph.num_nodes()

    tail = num_nodes % num_parts
    num_nodes_subgraph = num_nodes // num_parts

    perm, ptr = torch.arange(num_nodes), torch.tensor([(num_nodes_subgraph+1)*(i) for i in range(tail)] + [(num_nodes_subgraph+1)*tail+num_nodes_subgraph*(i) for i in range(num_parts-tail)] + [num_nodes,])

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return perm, ptr

def random_subgraph(graph, num_parts, recursive = False, log = True):
    if log:
        t = time.perf_counter()
        print(f'Computing randomly partitioning with {num_parts} parts...',)

    num_nodes = graph.num_nodes()

    tail = num_nodes % num_parts
    num_nodes_subgraph = num_nodes // num_parts

    perm, ptr = torch.randperm(num_nodes), torch.tensor([(num_nodes_subgraph+1)*(i) for i in range(tail)] + [(num_nodes_subgraph+1)*tail+num_nodes_subgraph*(i) for i in range(num_parts-tail)] + [num_nodes,])

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return perm, ptr

def metis(graph, num_parts: int, log: bool = True) -> List[Tensor,]:
    if log:
        t = time.perf_counter()
        print(f'Computing METIS partitioning with {num_parts} parts...',)
              
    num_nodes = graph.num_nodes()
    if num_parts <= 1:
        perm, ptr = torch.arange(num_nodes), torch.tensor([0, num_nodes])
    else:
        p_gs = dgl.metis_partition(graph, num_parts,)

        perm = []
        num_nodes_li = [0,]
        for k, val in p_gs.items():
            # nids = F.asnumpy(nids)
            perm.append(val.ndata[dgl.NID])
            num_nodes_li.append(val.num_nodes())
        perm = torch.cat(perm, dim=0)

        # ptr = torch.ops.torch_sparse.ind2ptr(cluster, num_parts)
        num_nodes_li = torch.LongTensor(num_nodes_li)
        ptr = num_nodes_li.clone()
        for i in range(len(num_nodes_li)-2):
            ptr[i+2:] = ptr[i+2:] + num_nodes_li[1:-(i+1)]

    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return perm, ptr


def permute(graph, perm: Tensor, log: bool = True):
    if log:
        t = time.perf_counter()
        print('Permuting data...',)

    rg = dgl.subgraph.node_subgraph(graph, perm, store_ids=False)
    
    if log:
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    return rg


