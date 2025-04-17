from torch import Tensor
from typing import List
import os
import os.path as osp
from .subgraph_loader import *
from tqdm import tqdm
import dgl
import dgl.function as fn


# https://github.com/pytorch/pytorch/issues/117122


def ortho_basis(M):
    Q, _ = torch.linalg.qr(M)
    return Q


class TOPDataSimple(torch.utils.data.Dataset):
    def __init__(
        self,
        graph, perm, batches, batch_size, sample_config,
        num_layers,
        histories = None,
        save_dir = None,
        edge_dropout=0.0,
    ):
        self.sample_config = sample_config
        self.num_layers = num_layers

        self.perm = perm
        self.inv_perm = torch.zeros_like(perm)
        self.inv_perm[perm] = torch.arange(len(perm))

        max_batch_size = min([len(batch) for batch in batches])
        max_dimention = histories[-1].shape[-1]

        max_layer = max((max_batch_size // max_dimention) - 2, 1)

        init_features = torch.cat(histories[:max_layer+1], dim=-1)

        print(init_features.shape)
        self.save_dir = save_dir

        self.edge_dropout = edge_dropout
        self.batches = batches
        self.graph = graph

        self.use_gpu = True
        self._proprocess(batches, init_features)

    def compensate_top(self, subgraph, n_id, adj_compensate_all, Q_all,):
        batch_size = n_id.numel()
        
        inb_id = torch.randperm(batch_size)[:max(self.sample_config.inb_size, Q_all.shape[1])]

        with subgraph.local_scope():
            subgraph.ndata['feat0'] = Q_all[n_id].to(self.graph.device) # cuda()

            
            Q = subgraph.ndata['feat0'][inb_id].pinverse().T
            
            subgraph.update_all(fn.u_mul_e('feat0', 'weight', 'm'), fn.sum('m', 'feat0'))
            adj_t_out_sketch = adj_compensate_all[n_id].to(self.graph.device) - subgraph.ndata.pop('feat0').to(self.graph.device)
  
        return adj_t_out_sketch, Q, inb_id

    
    def compute_subgraph(self, n_ids, adj_compensate_all, Q_all):
        n_id = n_ids

        subgraph = self.graph.subgraph(n_id.to(self.graph.device))
        
        adj_t_sketch_compensate, sketch_codebook, inb_id = self.compensate_top(subgraph, n_id, adj_compensate_all, Q_all,)



        subgraph.adj_t_sketch_compensate = adj_t_sketch_compensate
        subgraph.sketch_codebook = sketch_codebook
        subgraph.inb_id = inb_id
        
        subgraph.n_id = n_id
        subgraph.offset = torch.LongTensor([n_id[0]])
        subgraph.count = torch.LongTensor([n_id[-1] - n_id[0] + 1])
        
        if 'weight' in subgraph.edata:
            subgraph.edata['weight']

        return subgraph 
    
    def compute_q_adj_global(self, features):
        features = features.to(self.graph.device)
        O = torch.randn(features.shape[1], max(self.sample_config.rank, features.shape[1]), device=features.device) # full rank
        features = features @ O
        features = ortho_basis(features).to(self.graph.device)
        with self.graph.local_scope():
            self.graph.ndata['feat0'] = features
            self.graph.update_all(fn.u_mul_e('feat0', 'weight', 'm'), fn.sum('m', 'feat0'))
            adj_compensate_all = self.graph.ndata.pop('feat0')
            return adj_compensate_all, features


    def _proprocess(self, batches, features):


        self.data_list = []
        adj_compensate_all, Q_all = self.compute_q_adj_global(features)

        for i in tqdm(range(len(batches))):
            self.data_list.append(self.compute_subgraph(batches[i], adj_compensate_all, Q_all))


    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx] # .clone()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self.data_list)})'
    

class TOPDataOBSimple(TOPDataSimple):
    def __init__(
        self,
        graph, perm, batches, batch_size, sample_config,
        num_layers,
        histories = None,
        save_dir = None,
        edge_dropout=0.0,
    ):
        self.histories = histories
        super().__init__(graph, perm, batches, batch_size, sample_config,
            num_layers,
            histories,
            save_dir,
            edge_dropout)

        


    def compensate_top(self, subgraph, n_id, Q_all,):
        batch_size = n_id.numel()
        
        inb_id = torch.randperm(batch_size)[:max(self.sample_config.inb_size, Q_all.shape[1])]
        if self.use_gpu:

            adj_t_out_sketch = Q_all[subgraph.srcdata[dgl.NID][subgraph.num_dst_nodes():]].cuda()
            Q = Q_all[n_id[inb_id]].cuda().pinverse().T

            return adj_t_out_sketch, Q, inb_id
        else:
            adj_t_out_sketch = Q_all[subgraph.srcdata[dgl.NID][subgraph.num_dst_nodes():]]
            Q = Q_all[n_id[inb_id]].pinverse().T

            return adj_t_out_sketch, Q, inb_id
    
    def compute_subgraph(self, n_ids, Q_all):
        n_id = n_ids

        n_id = n_id.to(self.graph.device)
        frontier = dgl.sampling.sample_neighbors(self.graph, n_id, -1,)
        seed_nodes = dgl.utils.prepare_tensor(self.graph, n_id, 'items')
        subgraph = dgl.to_block(frontier, seed_nodes)

        adj_t_sketch_compensate, sketch_codebook, inb_id = self.compensate_top(subgraph, n_id, Q_all,)

        
        subgraph.adj_t_sketch_compensate = adj_t_sketch_compensate
        subgraph.sketch_codebook = sketch_codebook
        subgraph.inb_id = inb_id
        
        subgraph.n_id = n_id
        subgraph.offset = torch.LongTensor([n_id[0]])
        subgraph.count = torch.LongTensor([n_id[-1] - n_id[0] + 1])

        return subgraph 

    def compute_q_adj_global(self, features):
        O = torch.randn(features.shape[1], max(self.sample_config.top_k, features.shape[1]), device=features.device)
        Y = features @ O
        Q_all = ortho_basis(Y)

        Q_all = Q_all.to(self.graph.device)

        return Q_all


    def _proprocess(self, batches, init_features):
        self.init_features = init_features
        
        self.data_list = []

        Q_all = self.compute_q_adj_global(init_features)


        for i in tqdm(range(len(batches))):
            self.data_list.append(self.compute_subgraph(batches[i], Q_all))
        
        return self.data_list


