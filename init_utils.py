import argparse
import json
import math
import pathlib
import pickle
import shutil
import time
import traceback

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from CHANGELOG import MODEL_VERSION
from tiger.data.data_loader import ChunkSampler, GraphCollator, load_jodie_data
from tiger.data.graph import Graph
from tiger.eval_utils import eval_edge_prediction, warmup
from tiger.model.feature_getter import NumericalFeature
from tiger.model.restarters import SeqRestarter, StaticRestarter
from tiger.model.tiger import TIGER
from tiger.utils import BackgroundThreadGenerator
from train_utils import EarlyStopMonitor, get_logger, hash_args, seed_all


def init_parser():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('-d', '--data', type=str, default='wikipedia', help='Dataset name')
    parser.add_argument('--root', type=str, default='.', help='Dataset root')
    parser.add_argument('--dim', type=int, default=None, help='Feature dimension')
    parser.add_argument('--no_feat_buffer', action='store_true', 
                        help='Do not pre-load data into GPU')
    # Model
    parser.add_argument('--n_layers', type=int, default=1, help='Number of graph layers')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Number of temporal neighbors')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of graph heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--strategy', type=str, default="recent_edges",
                        choices=["recent_nodes", "recent_edges", "uniform"],
                        help='Sampling strategy for temporal aggregation')
    parser.add_argument('--msg_src', type=str, default="left", choices=["left", "right"],
                        help='message source')
    parser.add_argument('--upd_src', type=str, default="right", choices=["left", "right"],
                        help='update source')
    parser.add_argument('--upd_fn', type=str, default="gru", choices=["merge", "gru"],
                        help='memory update function')
    parser.add_argument('--tsfm_fn', type=str, default="id", choices=["id", "linear", "mlp"],
                        help='message transform function')
    parser.add_argument('--hit_type', type=str, default="bin",
                        choices=["vec", "bin", "count", "none"], help='Hit type')
    # -- restarter
    parser.add_argument('--mutual_coef', type=float, default=1.0, help='Mutual loss coef')
    parser.add_argument('--restart_prob', type=float, default=0.0, help='restart probability (train)')
    parser.add_argument('--restarter_type', type=str, default="seq",
                        choices=["seq", "static"], help='Restarter type')
    parser.add_argument('--hist_len', type=int, default=40, help='Length of history')
    parser.add_argument('--warmup', type=int, default=0, help='Number of warmup steps')

    return parser



def init_data(data, root, seed, rank=None, world_size=None, *,
              num_workers, bs, warmup_steps,
              subset, strategy,
              n_layers, n_neighbors,
              restarter_type, hist_len):

    (
        nfeats, efeats, shift_max, full_data, train_data, val_data, test_data,
        inductive_val_data, inductive_test_data
    ) = load_jodie_data(data, train_seed=seed, root=root)
    if subset < 1.0:
        train_end_id = math.ceil(len(train_data) * subset)
        offline_data = train_data.get_subset(train_end_id, len(train_data))
        train_data = train_data.get_subset(0, train_end_id)
    train_graph = Graph.from_data(train_data, strategy=strategy, seed=seed)
    full_graph = Graph.from_data(full_data, strategy=strategy, seed=seed)

    train_collator = GraphCollator(train_graph, n_neighbors, n_layers,
                                   restarter=restarter_type, hist_len=hist_len)
    eval_collator = GraphCollator(full_graph, n_neighbors, n_layers,
                                  restarter=restarter_type, hist_len=hist_len)

    if world_size is not None:
        train_sampler = ChunkSampler(len(train_data), rank=rank, world_size=world_size, bs=bs, seed=seed)
        train_dl = DataLoader(
            train_data, batch_size=bs, sampler=train_sampler,
            collate_fn=train_collator, pin_memory=True
        )  # distributed Dataloader
        offline_dl = None
    else:
        train_dl = DataLoader(train_data, batch_size=bs, collate_fn=train_collator, pin_memory=True, num_workers=num_workers)
        if subset < 1.0:
            offline_dl = DataLoader(offline_data, batch_size=bs, collate_fn=eval_collator)
        else:
            offline_dl = None

    val_dl = DataLoader(val_data, batch_size=bs, collate_fn=eval_collator)
    ind_val_dl = DataLoader(inductive_val_data, batch_size=bs, collate_fn=eval_collator)
    test_dl = DataLoader(test_data, batch_size=bs, collate_fn=eval_collator)
    ind_test_dl = DataLoader(inductive_test_data, batch_size=bs, collate_fn=eval_collator)
    
    if warmup_steps > 0:
        if len(train_data) - warmup_steps < 0 or len(val_data) - warmup_steps < 0:
            raise ValueError('Too many warmup steps!')

        val_warmup_data = train_data.get_subset(len(train_data) - warmup_steps, len(train_data))
        test_warmup_data = val_data.get_subset(len(val_data) - warmup_steps, len(val_data))
        val_warmup_dl = DataLoader(val_warmup_data, batch_size=bs, collate_fn=train_collator)
        test_warmup_dl = DataLoader(test_warmup_data, batch_size=bs, collate_fn=eval_collator)
    else:
        val_warmup_dl = test_warmup_dl = None

    basic_data = (nfeats, efeats, shift_max, full_data, train_data, val_data, test_data,
                  inductive_val_data, inductive_test_data)
    graphs = (train_graph, full_graph)
    dls = (train_dl, offline_dl, val_dl, ind_val_dl, test_dl, ind_test_dl, val_warmup_dl, test_warmup_dl)

    return basic_data, graphs, dls




def init_model(nfeats, efeats, train_graph, full_graph, full_data, device,
               *,
               feature_as_buffer, dim, n_layers, n_heads, n_neighbors, hit_type, dropout,
               restarter_type, hist_len, msg_src, upd_src, msg_tsfm_type, mem_update_type, imporatance_dim,
               shift_max, threshold, adl_control):

    if nfeats is not None:
        nfeats = torch.from_numpy(nfeats).float()
        dim = nfeats.shape[1] if dim is None else dim
    if efeats is not None:
        efeats = torch.from_numpy(efeats).float()
        dim = efeats.shape[1] if dim is None else dim

    raw_feat_getter = NumericalFeature(
        nfeats, efeats, dim=dim, register_buffer=feature_as_buffer, device=device
    )
    raw_feat_getter.n_nodes = full_graph.num_node
    raw_feat_getter.n_edges = len(full_data)

    if restarter_type == 'seq':
        restarter = SeqRestarter(
            raw_feat_getter=raw_feat_getter,
            graph=train_graph,
            hist_len=hist_len,
            n_head=n_heads, dropout=dropout
        )
    elif restarter_type == 'static':
        restarter = StaticRestarter(
            raw_feat_getter=raw_feat_getter,
            graph=train_graph
        )
    else:
        raise NotImplementedError
    model = TIGER(
        raw_feat_getter=raw_feat_getter, graph=train_graph, 
        restarter=restarter,
        n_neighbors=n_neighbors,
        hit_type=hit_type,
        n_layers=n_layers, n_head=n_heads, dropout=dropout,
        msg_src=msg_src, upd_src=upd_src,
        msg_tsfm_type=msg_tsfm_type, mem_update_type=mem_update_type,
        tgn_mode=True, msg_last_only=True, importance_dim=imporatance_dim, shift_max=shift_max,
        threshold=threshold, adl_control=adl_control
    ).to(device)
    return model