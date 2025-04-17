import time
import hydra
from omegaconf import OmegaConf
import yaml
import traceback
import os
import json

import numpy as np
import torch
import dgl

from torch.utils.data import DataLoader
from dgl_autoscale import (get_data, metis, random_subgraph, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                       models, compute_micro_f1, compute_auc, dropout)
from dgl_autoscale.loader.top_loader import * # , TOPData_large

torch.manual_seed(123)

def mini_train(model, loader, criterion, optimizer, max_steps, grad_norm=None,
               transform_dropedge = None):
    model.train()

    total_loss = total_examples = 0
    for i, subgraph in enumerate(loader):

        subgraph = subgraph.to(model.device)
        batch_feat = subgraph.srcdata['feat']
        y = subgraph.dstdata['label']
        train_mask = subgraph.dstdata['train_mask']
        
        if subgraph.adj_t_sketch_compensate is not None:
            adj_t_sketch_compensate = subgraph.adj_t_sketch_compensate.to(model.device)
            sketch_codebook = subgraph.sketch_codebook.to(model.device)
        else:
            adj_t_sketch_compensate = sketch_codebook = None

        if train_mask.sum() == 0:
            print('num_minitrain = 0')
            continue

        # We make use of edge dropout on ogbn-products to avoid overfitting.
        if transform_dropedge is not None:
            # subgraph = transform_dropedge(subgraph)
            subgraph = dropout(subgraph, transform_dropedge)

        optimizer.zero_grad()
        out = model(subgraph, batch_feat, adj_t_sketch_compensate, sketch_codebook, subgraph.inb_id)
        loss = criterion(out[train_mask], y[train_mask])


        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())

        # if (i + 1) >= 10:
        #     break
        # We may abort after a fixed number of steps to refresh histories...
        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break

    return total_loss / total_examples


@torch.no_grad()
def full_test(model, fulldata_loader):
    model.eval()


    outs = model.mini_inference_serial(fulldata_loader).cpu()
    return outs

@torch.no_grad()
def mini_test(model, loader):
    model.eval()
    total_acc_val = total_val_examples = 0
    total_acc_test = total_test_examples = 0
    for i, subgraph in enumerate(loader): # (batch, batch_size, *args)
        subgraph = subgraph.to(model.device)

        
        batch_feat = subgraph.srcdata['feat']
        

        # We make use of edge dropout on ogbn-products to avoid overfitting.
        if subgraph.adj_t_sketch_compensate is not None:
            adj_t_sketch_compensate = subgraph.adj_t_sketch_compensate.to(model.device)
            sketch_codebook = subgraph.sketch_codebook.to(model.device)
        else:
            adj_t_sketch_compensate = sketch_codebook = None


        out = model(subgraph, batch_feat, adj_t_sketch_compensate, sketch_codebook, subgraph.inb_id)
        if subgraph.dstdata['label'].shape[-1] == 1:
            if subgraph.dstdata['val_mask'].sum() > 0:
                acc_val = compute_auc(out, subgraph.dstdata['label'], subgraph.dstdata['val_mask'])
            else:
                acc_val = 0
            if subgraph.dstdata['test_mask'].sum() > 0:
                acc_test = compute_auc(out, subgraph.dstdata['label'], subgraph.dstdata['test_mask'])
            else:
                acc_test = 0
        else:
            if subgraph.dstdata['val_mask'].sum() > 0:
                acc_val = compute_micro_f1(out, subgraph.dstdata['label'], subgraph.dstdata['val_mask'])
            else:
                acc_val = 0
            if subgraph.dstdata['test_mask'].sum() > 0:
                acc_test = compute_micro_f1(out, subgraph.dstdata['label'], subgraph.dstdata['test_mask'])
            else:
                acc_test = 0

        
        val_examples = subgraph.dstdata['val_mask'].sum().item()
        total_acc_val += float(acc_val) * int(val_examples)
        total_val_examples += int(val_examples)

        test_examples = subgraph.dstdata['test_mask'].sum().item()
        total_acc_test += float(acc_test) * int(test_examples)
        total_test_examples += int(test_examples)

    return total_acc_val/total_val_examples, total_acc_test/total_test_examples

def get_top_dataset_partition(conf, params, graph, perm, ptr, dataset_processed_dir, histories):
    if conf.model.partition in ['metis', 'random_subgraph']:
        ptr = ptr[::params.partition_config.batch_size]
        if int(ptr[-1]) != graph.num_nodes():
            ptr = torch.cat([ptr, torch.tensor([graph.num_nodes()], dtype=ptr.dtype)], dim=0)
        n_id = torch.arange(graph.num_nodes(), dtype=ptr.dtype)
        if conf.model.use_eval_data:
            batches_train = n_id.split((ptr[1:] - ptr[:-1]).tolist())
            # batches_eval = n_id.split((ptr[1:] - ptr[:-1]).tolist())
            # batch_size = params['batch_size'] # if conf.dataset.name in ['products'] else int(0.25*params.partition_config.num_parts) 
            top_train_data = eval(conf.model.sampler_name)(graph, perm, batches_train, batch_size=params.partition_config.batch_size,
                                        sample_config=params.compensate_sample_config,
                                        num_layers=params.architecture.num_layers,
                                        histories=histories,
                                        save_dir=dataset_processed_dir,
                                        edge_dropout = params.edge_dropout,
                                        ).data_list
            top_eval_data = top_train_data
        else:
            batches_train = []
            for batch_n_id in list(n_id.split((ptr[1:] - ptr[:-1]).tolist())):
                if graph.ndata['train_mask'].cpu()[batch_n_id].sum() != 0:
                    batches_train.append(batch_n_id[graph.ndata['train_mask'].cpu()[batch_n_id]])
            batches_eval = []
            for batch_n_id in list(n_id.split((ptr[1:] - ptr[:-1]).tolist())):
                if (~graph.ndata['train_mask'][batch_n_id]).sum() != 0:
                    batches_eval.append(batch_n_id[~graph.ndata['train_mask'].cpu()[batch_n_id]])
            
            batches = batches_train + batches_eval

            # batch_size = params['batch_size'] # if conf.dataset.name in ['products'] else int(0.25*params.partition_config.num_parts) 
            subgraphs = eval(conf.model.sampler_name)(graph, perm, batches, batch_size=params.partition_config.batch_size,
                                        sample_config=params.compensate_sample_config,
                                        num_layers=params.architecture.num_layers,
                                        histories=histories,
                                        save_dir=dataset_processed_dir,
                                        edge_dropout = params.edge_dropout,
                                        ).data_list
            top_train_data = subgraphs[:len(batches_train)]
            top_eval_data = subgraphs[len(batches_train):]
    else:
        raise NotImplementedError
    return top_train_data, top_eval_data


def compute_buffer_size(eval_loader):
    buffer_size_list = []
    for _, _, n_id, _, _ in eval_loader:
        buffer_size_list.append(n_id.numel())
    buffer_size = max(buffer_size_list)
    return buffer_size


def get_gnn_loader(conf, params, model, fulldata_loader, graph, perm, ptr, dataset_processed_dir, histories):
    if histories is None:
        first_hist = True
        t = time.perf_counter()
        print('full_test...',)
        histories = []
        for i in tqdm(range(params.compensate_sample_config.random_num)):
            model.reset_parameters()
            full_test(model, fulldata_loader)
            histories = histories + [hist.emb.clone() for hist in model.histories]
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    else:
        first_hist = False
        t = time.perf_counter()
        print('full_test...',)
        full_test(model, fulldata_loader)
        histories = histories + [hist.emb.clone() for hist in model.histories]
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    print('top_data...',)

    t = time.perf_counter()
    model.eval()
    with torch.no_grad():
        initial_feature = model.initial_feature(graph.ndata['feat']).cpu()
    if histories is None:
        histories = initial_feature
    else:
        if first_hist:
            histories = [initial_feature,] + histories
    top_train_data, top_eval_data = get_top_dataset_partition(conf, params, graph, perm, ptr, dataset_processed_dir, histories)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    train_loader = DataLoader(top_train_data, batch_size=1,
                            collate_fn=lambda x: x[0],
                            shuffle=True, num_workers=params.num_workers,
                            persistent_workers=params.num_workers > 0)
    eval_loader = DataLoader(top_eval_data, batch_size=1,
                            collate_fn=lambda x: x[0],
                            shuffle=False,)
    return train_loader, eval_loader, histories


@hydra.main(config_path='conf_top', config_name='config')
def main(conf):
    conf.model.params = conf.model.params[conf.dataset.name]

    if 'json' in conf.model and conf.model.json is not None:
        print('Json path does not exist!')
        exit()

    params = conf.model.params
    dict_conf = yaml.safe_load(OmegaConf.to_yaml(conf))
    print(dict_conf)
    try:
        edge_dropout = params.edge_dropout
    except:  # noqa
        edge_dropout = 0.0
    grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    device = f'cuda' if torch.cuda.is_available() else 'cpu'

    t = time.perf_counter()
    print('Loading data...',)
    graph, in_channels, out_channels, dataset_processed_dir = get_data(conf.dataset.name,)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')


    if conf.model.partition in ['metis', 'random_subgraph']:
        if dataset_processed_dir is not None:
            metis_processed_dir = os.path.join(dataset_processed_dir, conf.model.partition, str(params.partition_config.num_parts),)
            os.makedirs(metis_processed_dir, exist_ok=True)
            if conf.model.norm is not None:
                dataset_processed_dir = os.path.join(dataset_processed_dir, conf.model.norm, str(conf.model.loop),)
            else:
                dataset_processed_dir = os.path.join(dataset_processed_dir, 'null', str(conf.model.loop),)
        if dataset_processed_dir is not None and os.path.exists(os.path.join(metis_processed_dir, 'perm.pt')) and os.path.exists(os.path.join(metis_processed_dir, 'ptr.pt')):
            perm = torch.load(os.path.join(metis_processed_dir, 'perm.pt'))
            ptr = torch.load(os.path.join(metis_processed_dir, 'ptr.pt'))
        else:
            perm, ptr = eval(conf.model.partition)(graph, num_parts=params.partition_config.num_parts, log=True)
            if dataset_processed_dir is not None:
                torch.save(perm, os.path.join(metis_processed_dir, 'perm.pt'))
                torch.save(ptr, os.path.join(metis_processed_dir, 'ptr.pt'))
    else:
        raise NotImplementedError
    
    graph = permute(graph, perm, log=True)

    if torch.__version__ >= (2, 0):
        cast_to_int = max(graph.num_nodes(), graph.num_edges()) <= 2e9
        if cast_to_int:
            ptr = ptr.int() # .long()
            graph = graph.int() # .long()

    if conf.model.loop:
        t = time.perf_counter()
        print('Adding self-loops...',)
        graph = graph.remove_self_loop().add_self_loop()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    if conf.model.norm is not None:
        t = time.perf_counter()
        print('Normalizing data...',)
        if conf.model.norm in ['DAD']:
            norm = dgl.nn.EdgeWeightNorm(norm='both')
            graph.edata['weight'] = norm(graph, torch.ones(graph.num_edges()))
        elif conf.model.norm in ['DA']:
            norm = dgl.nn.EdgeWeightNorm(norm='left')
            graph.edata['weight'] = norm(graph, torch.ones(graph.num_edges()))
        elif conf.model.norm in ['AD']:
            norm = dgl.nn.EdgeWeightNorm(norm='right')
            graph.edata['weight'] = norm(graph, torch.ones(graph.num_edges()))
        elif conf.model.norm in ['A']:
            graph = graph
        else:
            raise NotImplementedError
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    label_cpu = graph.ndata['label']
    train_mask_cpu = graph.ndata['train_mask']
    val_mask_cpu = graph.ndata['val_mask']
    test_mask_cpu = graph.ndata['test_mask']
    if not conf.model.data_cpu:
        graph = graph.to(device)
        graph.ndata['feat']
        if 'weight' in graph.edata:
            graph.edata['weight']

    t = time.perf_counter()
    print('Fulldata Loader...',)
    fulldata_loader = EvalSubgraphLoader(graph, True, True, ptr, batch_size=params.partition_config.batch_size)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    t = time.perf_counter()
    print('Calculating buffer size...',)
    # We reserve a much larger buffer size than what is actually needed for
    # training in order to perform efficient history accesses during inference.
    buffer_size = compute_buffer_size(fulldata_loader)
    print(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')

    t = time.perf_counter()
    print('GNN Model...',)
    GNN = getattr(models, conf.model.name)
    model = GNN(
        num_nodes=graph.num_nodes(),
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=params.pool_size,
        buffer_size=buffer_size,
        # buffer_size_train=buffer_size_train,
        # buffer_size_eval=buffer_size_eval,
        **params.architecture,
        # **kwargs,
    ).to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    
    if graph.ndata['label'].dim() == 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    histories = None

    kwargs = {}
    if conf.model.name[:3] == 'PNA':
        kwargs['deg'] = graph.in_degrees().float().clamp(min=1)

    transform_dropedge = dgl.DropEdge(edge_dropout)
    torch.cuda.reset_peak_memory_stats()

    results = torch.zeros(params.runs)
    try:
        for run in range(params.runs):
            optimizer = eval('torch.optim.'+params.optimizer_name)(model.parameters(), lr=params.lr, weight_decay=params.reg_weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                factor=params['lr_reduce_factor'],
                                patience=params['lr_schedule_patience'],
                                verbose=True)

            best_val_acc = test_acc = 0

            for epoch in range(1, int(params.epochs*conf.log_every) + 1):
                
                if epoch % 100000 == 1:
                    train_loader, eval_loader, histories = get_gnn_loader(conf, params, model, fulldata_loader, graph, perm, ptr, dataset_processed_dir, histories)

                loss = mini_train(model, train_loader, criterion, optimizer,
                                params.max_steps, grad_norm, transform_dropedge)
                
                if conf.model.full_graph_inference:
                    outs = full_test(model, fulldata_loader)

                    val_acc = compute_micro_f1(outs, label_cpu, val_mask_cpu)
                    tmp_test_acc = compute_micro_f1(outs, label_cpu, test_mask_cpu)
                else:
                    val_acc, tmp_test_acc = mini_test(model, eval_loader)
                    

                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    results[run] = test_acc
                    if conf.save_model is not None:
                        torch.save(model.state_dict(), os.path.join(conf.save_model, conf.model.name+'_'+conf.dataset.name+'.pt'))
                if epoch % conf.log_every == 0:
                    print(f'Epoch: {int(epoch // conf.log_every):04d}, Loss: {loss:.4f}, '
                        f'Val: {val_acc:.4f}, '
                        f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')

            model.reset_parameters()
            histories = None
        print(torch.mean(results), torch.std(results))
    except Exception as e:
            print(traceback.format_exc())



if __name__ == "__main__":
    main()
