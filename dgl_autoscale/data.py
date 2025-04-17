from typing import Tuple

import os
import torch
from torch import Tensor
import dgl
from dgl.data import citation_graph as citegrh
from dgl.data import reddit, yelp
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np

from .utils import index2mask, gen_masks


def normalize_gaussian(x):
    mean = x.mean(dim=0)
    std = 1.0/x.std(dim=0)
    std[std.isinf()] = 0
    return (x - mean) * std


def get_planetoid( name: str):
    if name.lower() == 'cora':
        dataset = citegrh.load_cora()
    elif name.lower() == 'citeseer':
        dataset = citegrh.load_citeseer()
    elif name.lower() == 'pubmed':
        dataset = citegrh.load_pubmed()
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
    return dataset[0], dataset[0].ndata['feat'].shape[1], dataset.num_classes, os.path.join(dataset.root, 'processed')


def get_arxiv(root, discretization=None):
    dataset = DglNodePropPredDataset('ogbn-arxiv', os.path.join(root, 'ogb'),)
    data = dataset[0]
    graph, label = data

    graph = dgl.to_bidirected(graph, copy_ndata=True)

    graph.ndata.pop('year')

    
    graph.ndata['feat'] = normalize_gaussian(graph.ndata['feat'])

    label = label.view(-1)
    split_idx = dataset.get_idx_split()

    graph.ndata['train_mask'] = index2mask(split_idx['train'], graph.num_nodes())
    graph.ndata['val_mask'] = index2mask(split_idx['valid'], graph.num_nodes())
    graph.ndata['test_mask'] = index2mask(split_idx['test'], graph.num_nodes())

    graph.ndata['label'] = label

    return graph, graph.ndata['feat'].shape[1], dataset.num_classes, os.path.join(dataset.root, 'processed')


def get_papers(root, discretization=None):
    dataset = DglNodePropPredDataset('ogbn-papers100M', os.path.join(root, 'ogb'),)
    data = dataset[0]
    graph, label = data

    graph = dgl.to_bidirected(graph, copy_ndata=True)

    
    graph.ndata['feat'] = normalize_gaussian(graph.ndata['feat'])

    label = label.view(-1)
    split_idx = dataset.get_idx_split()

    graph.ndata['train_mask'] = index2mask(split_idx['train'], graph.num_nodes())
    graph.ndata['val_mask'] = index2mask(split_idx['valid'], graph.num_nodes())
    graph.ndata['test_mask'] = index2mask(split_idx['test'], graph.num_nodes())

    graph.ndata['label'] = label


    return graph, graph.ndata['feat'].shape[1], dataset.num_classes, os.path.join(dataset.root, 'processed')

def get_products(root, discretization=None):
    dataset = DglNodePropPredDataset('ogbn-products', os.path.join(root, 'ogb'),)
    data = dataset[0]
    graph, label = data

    graph.ndata['feat'] = normalize_gaussian(graph.ndata['feat']) # , cluster_labels

    label = label.view(-1)
    split_idx = dataset.get_idx_split()
    
    graph.ndata['train_mask'] = index2mask(split_idx['train'], graph.num_nodes())
    graph.ndata['val_mask'] = index2mask(split_idx['valid'], graph.num_nodes())
    graph.ndata['test_mask'] = index2mask(split_idx['test'], graph.num_nodes())

    graph.ndata['label'] = label
    
    return graph, graph.ndata['feat'].shape[1], dataset.num_classes, os.path.join(dataset.root, 'processed')


def get_yelp(root, discretization=None):
    dataset = yelp.YelpDataset()
    graph = dataset[0]
    graph.ndata['feat'] = (graph.ndata['feat'] - graph.ndata['feat'].mean(dim=0)) / graph.ndata['feat'].std(dim=0)
    graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
    graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
    graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
    graph.ndata['label'] = graph.ndata['label'].float()
    return graph, graph.ndata['feat'].shape[1], dataset.num_classes, os.path.join(dataset.save_path,)


def get_reddit(root, discretization=None):
    dataset = reddit.RedditDataset()
    graph = dataset[0]
    graph.ndata['feat'] = (graph.ndata['feat'] - graph.ndata['feat'].mean(dim=0)) / graph.ndata['feat'].std(dim=0)
    return graph, graph.ndata['feat'].shape[1], dataset.num_classes, os.path.join(dataset.save_path,)



def get_data( name: str, discretization=None):
    root = os.path.expanduser('~/datasets')
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(name, root, discretization)
    elif name.lower() == 'reddit':
        return get_reddit(root, discretization)
    elif name.lower() == 'yelp':
        return get_yelp(root, discretization)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root, discretization)
    elif name.lower() in ['ogbn-products', 'products']:
        return get_products(root, discretization)
    elif name.lower() in ['ogbn-papers100m', 'papers']:
        return get_papers(root, discretization)
    else:
        raise NotImplementedError
