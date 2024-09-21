from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
from torch.nn import functional as F

from ..data.data_loader import ComputationGraph
from ..data.graph import Graph
from .basic_modules import MergeLayer
from .feature_getter import NumericalFeature
from .time_encoding import TimeEncode


class GraphEmbedding(nn.Module):

    def __init__(self, raw_feat_getter: NumericalFeature, time_encoder: TimeEncode, graph: Graph, n_neighbors=20, n_layers=2,
                 importance_dim: int = 100, shift_max: float = 0.):
        super().__init__()
        self.raw_feat_getter = raw_feat_getter
        self.time_encoder = time_encoder
        self.graph = graph
        self.n_neighbors = n_neighbors
        self.n_layers = n_layers
        self.importance_dim = importance_dim
        self.importance_encoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.importance_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.importance_dim, out_features=self.importance_dim))

        self.attentions = None
        self.hidden = None
        self.cell = None
        self.update_prob = None
        self.cum_update_prob = None

    @property
    def device(self):
        return self.raw_feat_getter.device


    def init_para_for_one_depth(self, batch_size, time_feat_dim, node_feat_dim):
        self.hidden = torch.zeros((1, batch_size, 4 * (node_feat_dim + time_feat_dim))).to(self.device)
        self.cell = torch.zeros((1, batch_size, 4 * (node_feat_dim + time_feat_dim))).to(self.device)
        self.update_prob = torch.ones(1).to(self.device)
        self.cum_update_prob = torch.zeros(1).to(self.device)

    def init_para_for_one_batch(self, ):
        self.attentions = [[] for _ in range(self.n_layers)]

    def compute_embedding_with_computation_graph(
        self, involved_node_reprs: Tensor, center_nids: Tensor, ts: Tensor,
        computation_graph: ComputationGraph, depth: Optional[int]=None
        ) -> Tensor:
        """
        Compute temporal embeddings h(t-) of center nodes at given timestamps
        using h(t'+) of involved nodes with computation graph
        -----
        involved_node_reprs: involved nodes' representations, Tensor, shape (batch_size, node_dim)
        center_nids: an 1d numpy array of center node ids. shape = [size]
        ts: an 1d numpy array of query timestamps. shape = [size]
        computation_graph: computation graph containing necessary info
        depth: the current depth
        -----
        Return: center_node_reprs
        center_node_reprs: a tensor of center node representations. shape (batch_size, node_dim)
        """
        depth = self.n_layers if depth is None else depth  # current depth, from n_layers to 0
        # global index -> local index of involved nodes
        # ndarray, shape (batch_size, )
        local_center_nids = computation_graph.local_index[center_nids]
        # Tensor, shape (batch_size, node_dim)
        center_node_reprs = involved_node_reprs[local_center_nids] \
                            + self.raw_feat_getter.get_node_embeddings(center_nids)
        if depth == 0:  # exit condition
            return center_node_reprs  # h(t'+) + static_feat

        n_involve_nodes, d = involved_node_reprs.shape
        # get neighbors directly from the computation graph
        # Get neighbors at the 'depth-th' layer
        #
        neigh_nids, neigh_eids, neigh_ts, neigh_re, neigh_degree = computation_graph.layers[depth]
        n_center, n_neighbors = neigh_nids.shape

        neigh_reprs = self.compute_embedding_with_computation_graph(
            involved_node_reprs=involved_node_reprs,
            center_nids=neigh_nids.flatten(),
            # ts=neigh_ts.flatten(), 
            ts=torch.repeat_interleave(ts, n_neighbors),  # TGN
            computation_graph=computation_graph,
            depth=depth-1)

        neigh_reprs = neigh_reprs.reshape(n_center, n_neighbors, d)
        edge_feats = self.raw_feat_getter.get_edge_embeddings(neigh_eids)  # 3D
        delta_ts = ts[:, None] - neigh_ts
        delta_ts_reprs = self.time_encoder(delta_ts)  # 3D
        t0_reprs = self.time_encoder(torch.zeros_like(delta_ts[:, 0]))  # 2D

        # Tensor, shape (batch_size, num_neighbors)
        importance_score = torch.div(neigh_degree, neigh_re)

        # mask nan
        nan_mask = neigh_re == 0
        importance_score = importance_score.masked_fill(nan_mask, 0.0)

        # shape (batch_size, num_neighbors, importance_dim)
        neigh_importance_features = self.importance_encoder(importance_score.unsqueeze(2).float().to(self.device))

        # multi-head attention
        self.init_para_for_one_depth(n_center, self.time_encoder.get_time_dim(), d)
        center_node_reprs, self.hidden, self.cell, attention, self.update_prob, self.cum_update_prob \
                          =self.aggregate(depth=depth,
                                           center_x=center_node_reprs,
                                           center_tx=t0_reprs, 
                                           neigh_x=neigh_reprs,
                                           edge_x=edge_feats,
                                           edge_tx=delta_ts_reprs, 
                                           mask=(neigh_nids == 0),
                                           neigh_imp=neigh_importance_features,
                                           time_inter=delta_ts,
                                           cell= self.cell,
                                           hidden=self.hidden,
                                           update_prob=self.update_prob,
                                           cum_update_prob=self.cum_update_prob
        )
        self.attentions[depth - 1].append(attention)

        return center_node_reprs
    
    def compute_embedding(
            self, all_node_reprs: Tensor, np_center_nids: np.ndarray, np_ts: np.ndarray, 
            depth: Optional[int]=None
        ) -> Tensor:
        """
        Compute temporal embeddings of center nodes at given timestamps.
        -----
        all_node_reprs: a tensor containing ALL nodes' representations
        center_nids: an 1d numpy array of center node ids. shape = [size]
        ts: an 1d numpy array of query timestamps. shape = [size]
        depth: the current depth
        -----
        Return: center_node_reprs
        center_node_reprs: a tensor of center node representations. shape = [size, D]
        """
        depth = self.n_layers if depth is None else depth  # current depth, from n_layers to 0

        center_nids = torch.from_numpy(np_center_nids).long().to(self.device)

        # temporal representations + static (transformed) features 
        center_node_reprs = all_node_reprs[center_nids] \
                            + self.raw_feat_getter.get_node_embeddings(center_nids)

        if depth == 0:  # exit condition
            return center_node_reprs  # h(t'+) + static_feat
        
        n_total, d = all_node_reprs.shape
        np_neigh_nids, np_neigh_eids, np_neigh_ts, *_ = self.graph.sample_temporal_neighbor(
            np_center_nids, np_ts, self.n_neighbors)  # inputs and outputs are all np.ndarray
        # filter and compress?
        if False:
            # remove all-padding columns. at least 1 column is kept to avoid bugs.
            np_neigh_nids, np_neigh_eids, np_neigh_ts = filter_neighbors(np_neigh_nids, np_neigh_eids, np_neigh_ts)
            
            n_center, n_neighbors = np_neigh_nids.shape  # maybe n_neighbors < self.n_neighbors for reducing computation

            # reduce repeat computation
            unique_neigh_nids, unique_neigh_ts, np_inverse_idx = compress_node_ts_pairs(
                np_neigh_nids.flatten(), np.repeat(np_ts, n_neighbors)
            )
            neigh_reprs = self.compute_embedding(all_node_reprs=all_node_reprs, 
                                                np_center_nids=unique_neigh_nids,
                                                np_ts=unique_neigh_ts,
                                                depth=depth-1)
            
            neigh_nids = torch.from_numpy(np_neigh_nids).long().to(self.device)
            neigh_eids = torch.from_numpy(np_neigh_eids).long().to(self.device)
            inverse_idx = torch.from_numpy(np_inverse_idx).long().to(self.device)

            neigh_reprs = neigh_reprs[inverse_idx]  # [n x n_neighbors, d]
        else:
            n_center, n_neighbors = np_neigh_nids.shape
            neigh_reprs = self.compute_embedding(all_node_reprs=all_node_reprs, 
                                                np_center_nids=np_neigh_nids.flatten(),
                                                # TODO: check this !!!!!!
                                                # np_ts=np_neigh_ts.flatten(),
                                                np_ts=np.repeat(np_ts, n_neighbors),  # TGN
                                                depth=depth-1)
            
            neigh_nids = torch.from_numpy(np_neigh_nids).long().to(self.device)
            neigh_eids = torch.from_numpy(np_neigh_eids).long().to(self.device)


        neigh_reprs = neigh_reprs.reshape(n_center, n_neighbors, d)
        edge_feats = self.raw_feat_getter.get_edge_embeddings(neigh_eids)  # 3D
        delta_ts = torch.from_numpy(np_ts[:, None] - np_neigh_ts).float().to(self.device)
        delta_ts_reprs = self.time_encoder(delta_ts)  # 3D
        t0_reprs = self.time_encoder(torch.zeros_like(delta_ts[:, 0]))  # 2D

        mask = neigh_nids == 0  # 2D

        center_node_reprs = self.aggregate(depth=depth,
                                           center_x=center_node_reprs,
                                           center_tx=t0_reprs, 
                                           neigh_x=neigh_reprs,
                                           edge_x=edge_feats,
                                           edge_tx=delta_ts_reprs, 
                                           mask=mask
        )

        return center_node_reprs

    def aggregate(self, depth: int, center_x: Tensor, center_tx: Tensor,
                  neigh_x: Tensor, edge_x: Tensor, edge_tx: Tensor,
                  mask: Tensor, neigh_imp: Tensor, time_inter: Tensor, cell: Tensor, hidden: Tensor,
                  update_prob: float, cum_update_prob: float):
        raise NotImplementedError


class GraphAttnEmbedding(GraphEmbedding):
    def __init__(self, raw_feat_getter: NumericalFeature, time_encoder: TimeEncode, graph: Graph, n_neighbors=20, n_layers=2, n_head=2, dropout=0.1,
                 importance_dim: int = 100, shift_max: float = 0.0):
        super().__init__(raw_feat_getter, time_encoder, graph, n_neighbors, n_layers, importance_dim, shift_max)
        self.n_head = n_head
        self.dropout = dropout
        self.fns = nn.ModuleList([TemporalAttention(
                nfeat_dim=self.raw_feat_getter.nfeat_dim,
                efeat_dim=self.raw_feat_getter.efeat_dim,
                tfeat_dim=self.time_encoder.dim,
                n_head=self.n_head, dropout=dropout,
                importance_dim=importance_dim,
                shift_max=shift_max
            ) for _ in range(self.n_layers)]
        )
    
    def aggregate(self, depth: int, center_x: Tensor, center_tx: Tensor,
                  neigh_x: Tensor, edge_x: Tensor, edge_tx: Tensor, 
                  mask: Tensor, neigh_imp: Tensor, time_inter: Tensor, cell: Tensor, hidden: Tensor,
                  update_prob: Tensor, cum_update_prob: Tensor):
        fn = self.fns[self.n_layers - depth]
        h, attention, hidden, cell, update_prob, cum_update_prob \
            = fn(node_features=center_x, node_time_features=center_tx,
               neighbor_node_features=neigh_x, neighbor_node_edge_features=edge_x, neighbor_node_time_features=edge_tx,
               neighbor_masks=mask, neighbor_node_importance_features=neigh_imp, time_interval=time_inter,
               cell=cell, hidden=hidden, update_prob=update_prob, cum_update_prob=cum_update_prob)
        return h, attention, hidden, cell, update_prob, cum_update_prob


class TemporalAttention(nn.Module):
    def __init__(self, nfeat_dim, efeat_dim, tfeat_dim, n_head=2, dropout=0.1, importance_dim=100, n_neighbors = 10,
                 shift_max=0.0):
        super().__init__()
        self.n_head = n_head
        self.dropout = dropout
        self.num_neighbors = n_neighbors
        self.query_dim = nfeat_dim + tfeat_dim
        self.key_dim = nfeat_dim + efeat_dim + tfeat_dim
        self.importance_score_dim = importance_dim + nfeat_dim + tfeat_dim + efeat_dim
        self.shift_max = shift_max

        assert self.query_dim % n_head == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // n_head

        self.query_projection = nn.Linear(self.query_dim, n_head * self.head_dim, bias=True)
        self.key_projection = nn.Linear(self.key_dim, n_head * self.head_dim, bias=True)
        self.value_projection = nn.Linear(self.key_dim, n_head * self.head_dim, bias=True)
        self.importance_projection = nn.Linear(self.importance_score_dim, n_head * self.head_dim, bias=True)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(n_head * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

        ### Importance score
        self.score_linear = nn.Linear(self.head_dim * self.num_neighbors, self.num_neighbors)

        # Res
        self.merger = MergeLayer(self.query_dim, nfeat_dim, nfeat_dim, nfeat_dim)

        # Depth Selector
        self.short_gate = nn.Sequential(nn.Linear(4 * self.query_dim, 4 * self.query_dim, bias=True), nn.Tanh())
        # self.linear = nn.Linear(1, hidden_size, bias=False)
        self.lstm = torch.nn.LSTM(self.query_dim, 4 * self.query_dim, 1, bias=True)
        self.linear = nn.Linear(4 * self.query_dim, self.query_dim, bias=True)

        self.linear_cell = nn.Linear(4 * self.query_dim, 4 * self.query_dim, bias=True)


    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor, neighbor_node_features: torch.Tensor,
            neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor, neighbor_masks: np.ndarray,
            neighbor_node_importance_features: torch.Tensor, time_interval: torch.Tensor, cell: torch.Tensor, hidden: torch.Tensor,
            update_prob: torch.Tensor, cum_update_prob: torch.Tensor) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        neighbor_node_importance_features, shape (batch size, num_neighbors, importance_dim)
        :return:
        """

        # shape (batch_size, num_neighbors, importance_dim + node_dim + time_dim + edge_dim)
        # print(neighbor_node_importance_features)  # 有nan
        importance_score = torch.tanh(torch.cat([neighbor_node_importance_features, neighbor_node_features, \
                                                 neighbor_node_time_features, neighbor_node_edge_features], dim=2))

        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)
        node_time_features = torch.unsqueeze(node_time_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.n_head, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.n_head, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.n_head, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        importance_score = self.importance_projection(importance_score).reshape(importance_score.shape[0],
                                                                                importance_score.shape[1], \
                                                                                self.n_head, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        importance_score = importance_score.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors * self.head_dim)
        importance_score = importance_score.reshape(importance_score.shape[0], importance_score.shape[1], \
                                                    1, importance_score.shape[2] * importance_score.shape[3])

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        importance_score = self.score_linear(importance_score)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = attention + importance_score

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        """ # Depth Selector
        # Tensor, shape (1, batch_size, node_dim + time_dim)
        cell_short = self.short_gate(cell)

        # Tensor, shape (batch_size, 1)
        time_interval = torch.mean(time_interval, dim=1).unsqueeze(1)  # 求平均
        # time_interval = torch.max(time_interval, dim=1).values.unsqueeze(1) # 求最大，离现在最远
        # time_interval = torch.min(time_interval, dim=1).values.unsqueeze(1)  # 求最小，离现在最近
        time_decay = torch.exp(- 2 * time_interval / self.shift_max)

        # Tensor, shape (1, batch_size, node_dim + time_dim)
        cell_new = cell - cell_short + cell_short.mul(time_decay)

        # Tensor, shape (1, batch_size, node_dim + time_dim)
        depth_input = output.unsqueeze(0)

        output, (hidden_output, cell_output) = self.lstm(depth_input, (hidden, cell_new))

        # Tensor, shape (batch_size, node_dim + time_dim)
        output = output.squeeze(0)
        output = self.linear(output)"""

        output, hidden_output, cell_output, update_prob, cum_update_prob\
            = self.skip_time_decay_depth_selector(input=output,
                                                  cell=cell,
                                                  hidden=hidden,
                                                  update_prob_prev=update_prob,
                                                  cum_update_prob_prev=cum_update_prob,
                                                  time_interval=time_interval)

        # Tensor, shape (batch_size, node_feat_dim)
        node_features = torch.squeeze(node_features, dim=1)
        # Res, Tensor, shape (batch_size, node_feat_dim)
        output = self.merger(output, node_features)

        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_for_neighbors = torch.mean(attention_scores, dim=1)

        return output, attention_for_neighbors, hidden_output, cell_output, update_prob, cum_update_prob

    def skip_time_decay_depth_selector(self, input: Tensor, cell: Tensor, hidden: Tensor,
                                       cum_update_prob_prev: Tensor, update_prob_prev: Tensor, time_interval:Tensor):
        """
        skip depth selector
        :param cell: tensor, shape (1, batch_size, hidden_dim)
        :param hidden: tensor, shape (1, batch_size, hidden_dim)
        :param time_interval: tensor, shape (batch_size, num_neighbors)
        :return:
        """
        # time decay
        # tensor, shape (1, batch_size, hidden_dim)
        cell_short = self.short_gate(cell)
        # tensor, shape (batch_size, 1)
        time_interval = torch.mean(time_interval, dim=1).unsqueeze(1)  # 求平均
        time_decay = torch.exp(- 2 * time_interval / self.shift_max)
        # tensor, shape (1, batch_size, node_dim + time_dim)
        cell_new = cell - cell_short + cell_short.mul(time_decay)

        # lstm
        # tensor, shape (1, batch_size, hidden_dim)
        depth_input = input.unsqueeze(0)
        output, (hidden_output, cell_output) = self.lstm(depth_input, (hidden, cell_new))

        # skip
        # compute value for the update prob
        new_update_prob_tilde = torch.sigmoid(self.linear_cell(hidden))  # tilde
        # compute value for the update gate
        cum_update_prob = cum_update_prob_prev + torch.min(update_prob_prev, 1. - cum_update_prob_prev)  # 这个batch原来的u_t
        # round
        bn = binarylayer()
        update_gate = bn.apply(cum_update_prob)
        # apply update gate
        new_c = update_gate * cell_output + (1. - update_gate) * cell_new
        new_h = update_gate * hidden_output + (1. - update_gate) * hidden
        # 如果更新了，则为delta；没更新，和上一个batch原来的u_t一样
        new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
        # 如果更新了，当前batch的现在的u_t清0；没更新，则为这个batch原来的u_t
        new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob

        # tensor, shape (batch_size, node_dim + time_dim)
        output = new_h.squeeze(0)
        output = self.linear(output)
        return output, new_h, new_c, new_update_prob, new_cum_update_prob


class BinaryLayer(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(result)
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        # result, = ctx.saved_tensors
        return grad_output


def filter_neighbors(ngh_nids: np.ndarray, ngh_eids: np.ndarray, ngh_ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop columns whose elements are all padding tokens.
    """
    col_mask = ~np.all(ngh_nids == 0, 0)  # if entire col is null, drop the col.
    col_mask[-1] = True  # at least have one (null) neighbor to aviod bugs
    ngh_nids = ngh_nids[:, col_mask]
    ngh_eids = ngh_eids[:, col_mask]
    ngh_ts = ngh_ts[:, col_mask]
    return ngh_nids, ngh_eids, ngh_ts


def compress_node_ts_pairs(nids: np.ndarray, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deduplicate node-ts pairs to reduce computations.
    """
    codes = np.stack([nids, ts])  # [2, layer_size]
    _, unique_idx, inverse_idx = np.unique(codes, axis=1, return_index=True, return_inverse=True)
    unique_nids = nids[unique_idx]
    unique_ts = ts[unique_idx]
    return unique_nids, unique_ts, inverse_idx
