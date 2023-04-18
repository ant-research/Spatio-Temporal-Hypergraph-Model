import numpy as np
from typing import List, Optional, NamedTuple
from scipy.sparse import coo_matrix
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean, scatter_max, scatter_add, scatter_min
from utils import haversine
import logging


class NeighborSampler(torch.utils.data.DataLoader):
    """
    Hypergraph sampler with two level hypergraph for next-poi task.

    Args:
        x: feature matrix.
        edge_index: two tensor-composited list, the first one is the edge_index of traj2traj(index 0), the second is the
            edge_index of ci2traj(index 1).
        edge_attr: traj2traj jaccard similarity, source hyperedge size and target hyperedge size.
        edge_t: actual time of each checkin event, so traj2traj(index 0) doesnt contain this edge_t.
        edge_delta_t: relative time within trajectory, traj2traj(index 0) doesnt contain this value.
        edge_type: intra-user(0) or inter-user(1) indicator, but ci2traj(index 1) doesnt contain this value.
        sizes: the last element is for ci2traj, other elements is for multi-hop traj2traj. e.g. sizes=[10, 20, 30],
            [10,20] is for traj2traj 2-hop sampling, [30] is for ci2traj.
        sample_idx: sample id, torch.long.
        node_idx: query trajectory id, torch.long.
        label: task label for loss computation, tensor with 4 columns (poi_id, poi_cat_id, poi_lat, poi_lon).
        edge_delta_s: relative distance within trajectory, traj2traj(index 0) doesnt contain this value
        max_time: target time of every sample, last checkin time before candidate checkin.
        num_nodes: max trajectory index. The number of nodes in the graph.
            (default: :obj:`None`)
        intra_jaccard_threshod: filter out intra-user traj2traj, when the jaccard similarity is less than this value.
        inter_jaccard_threshod: filter out inter-user traj2traj, when the jaccard similarity is less than this value.
        transform: A function/transform that takes in an a sampled mini-batch and returns a transformed version.
        **kwargs: Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """

    def __init__(
            self,
            x: List[Tensor],
            edge_index: List[Tensor],
            edge_attr: List[Tensor],
            edge_t: List[Tensor],
            edge_delta_t: List[Tensor],
            edge_type: List[Tensor],
            sizes: List[int],
            sample_idx: Tensor,
            node_idx: Tensor,
            label: Tensor,
            edge_delta_s: List[Tensor] = None,
            max_time: Optional[Tensor] = None,
            num_nodes: Optional[int] = None,
            intra_jaccard_threshold: float = 0.0,
            inter_jaccard_threshold: float = 0.0,
            **kwargs
    ):
        # raw feature
        traj_x = x[0]
        ci_x = x[1]

        # traj2traj related
        traj2traj_edge_attr = edge_attr[0]
        traj2traj_edge_index = edge_index[0]
        traj2traj_edge_type = edge_type[0]
        traj2traj_edge_delta_t = edge_delta_t[0]
        traj2traj_edge_delta_s = edge_delta_s[0]

        # ci2traj related
        ci2traj_edge_index = edge_index[1]
        ci2traj_edge_t = edge_t[1]
        ci2traj_edge_delta_t = edge_delta_t[1]
        ci2traj_edge_delta_s = edge_delta_s[1]

        # to cpu
        traj_x = traj_x.to('cpu')
        ci_x = ci_x.to('cpu')
        traj2traj_edge_attr = traj2traj_edge_attr.to('cpu')
        traj2traj_edge_index = traj2traj_edge_index.to('cpu')
        traj2traj_edge_type = traj2traj_edge_type.to('cpu')
        traj2traj_edge_delta_t = traj2traj_edge_delta_t.to('cpu')
        traj2traj_edge_delta_s = traj2traj_edge_delta_s.to('cpu')
        ci2traj_edge_index = ci2traj_edge_index.to('cpu')
        ci2traj_edge_t = ci2traj_edge_t.to('cpu')
        ci2traj_edge_delta_t = ci2traj_edge_delta_t.to('cpu')
        ci2traj_edge_delta_s = ci2traj_edge_delta_s.to('cpu')

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        self.traj_x = traj_x
        self.ci_x = ci_x
        self.max_traj_size = self.traj_x[:, 0].max().item()
        self.x = torch.cat([ci_x, traj_x], dim=0)
        self.ci_offset = ci_x.shape[0]
        self.traj2traj_edge_attr = traj2traj_edge_attr
        self.traj2traj_edge_type = traj2traj_edge_type
        self.traj2traj_edge_delta_t = traj2traj_edge_delta_t
        self.traj2traj_edge_delta_s = traj2traj_edge_delta_s
        self.ci2traj_edge_t = ci2traj_edge_t
        self.ci2traj_edge_delta_t = ci2traj_edge_delta_t
        self.ci2traj_edge_delta_s = ci2traj_edge_delta_s

        self.y = label  # load from data
        self.node_idx = node_idx  # target trajectory index, used as query
        self.max_time = max_time
        self.sizes = sizes
        self.he2he_jaccard = None
        self.intra_jaccard_threshold = intra_jaccard_threshold
        self.inter_jaccard_threshold = inter_jaccard_threshold

        # Obtain a *transposed* SparseTensor instance.
        if int(node_idx.max()) > traj2traj_edge_index.max():
            raise ValueError('Query node index is not in graph.')
        if num_nodes is None:
            num_nodes = max(int(traj2traj_edge_index.max()), int(ci2traj_edge_index.max())) + 1

        self.traj2traj_adj_t = SparseTensor(
            row=traj2traj_edge_index[0],
            col=traj2traj_edge_index[1],
            value=torch.arange(traj2traj_edge_index.size(1)),
            sparse_sizes=(num_nodes, num_nodes)
        ).t()
        self.ci2traj_adj_t = SparseTensor(
            row=ci2traj_edge_index[0],
            col=ci2traj_edge_index[1],
            value=torch.arange(ci2traj_edge_index.size(1)),
            sparse_sizes=(num_nodes, num_nodes)
        ).t()

        self.traj2traj_adj_t.storage.rowptr()
        self.ci2traj_adj_t.storage.rowptr()

        super(NeighborSampler, self).__init__(sample_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        """
        adjs: traj2traj multi-hop + ci2traj one-hop adj_t data

        :param batch:
        :return:
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        sample_idx = batch
        n_id = self.node_idx[sample_idx]
        max_time = self.max_time[sample_idx]

        # Sample traj2traj multi-hop dynamic relation
        for i, size in enumerate(self.sizes):
            # n_id is the original node_id, the row and col idx in adj_t is mapped to 0~(len(n_id)-1)
            if i == len(self.sizes) - 1:
                # Sample ci2traj one-hop checkin relation
                adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_t = self.ci2traj_edge_t[e_id]
                edge_type = None
                edge_delta_t = self.ci2traj_edge_delta_t[e_id]
                edge_delta_s = self.ci2traj_edge_delta_s[e_id]
            else:
                # Sample traj2traj multi-hop relation
                adj_t, n_id = self.traj2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = self.traj2traj_edge_attr[e_id]
                edge_t = None
                edge_type = self.traj2traj_edge_type[e_id]
                edge_delta_t = self.traj2traj_edge_delta_t[e_id]
                edge_delta_s = self.traj2traj_edge_delta_s[e_id]

            size = adj_t.sparse_sizes()[::-1]

            if adj_t.nnz():
                assert size[0] >= col.max() + 1, '[NeighborSampler] adj_t source max index exceed sparse_sizes[1]'
            else:
                # empty subgraph
                adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s = None, None, None, None, None, None
            adjs.append((adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size))

        # Mask ci2traj for target traj: filter only ci2traj edge beyond target time
        target_mask = row < batch_size
        edge_max_time = max_time[row[target_mask]]
        length = torch.sum(target_mask)
        time_mask = edge_t[target_mask] <= edge_max_time
        target_mask[:length] = time_mask

        if row[target_mask].size(0) == 0:
            raise ValueError(
                f'[NeighborSampler] All trajectories have no checkin before target time!!'
            )
        adj_t = SparseTensor(
           row=row[target_mask],
           col=col[target_mask],
           sparse_sizes=(batch_size, adj_t.sparse_sizes()[1])
        )
        edge_t = edge_t[target_mask]
        edge_type = None
        edge_delta_t = edge_delta_t[target_mask]
        edge_delta_s = edge_delta_s[target_mask]
        e_id = e_id[target_mask]
        adjs.append((adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size))

        # Filter traj2traj with leakage
        target_mask[length:] = True
        he_poi = self.ci_x[col[target_mask]][:, 1]
        im = coo_matrix((
            np.ones(row[target_mask].shape[0]),
            (he_poi.numpy().astype(np.long), row[target_mask].numpy())
        )).tocsr()
        self.he2he_jaccard = im.T * im
        self.he2he_jaccard = self.he2he_jaccard.tocoo()

        # Calculate jaccard similarity of traj2traj
        filtered_traj_size = self.he2he_jaccard.diagonal()
        source_size = filtered_traj_size[self.he2he_jaccard.col]
        target_size = filtered_traj_size[self.he2he_jaccard.row]
        self.he2he_jaccard.data = self.he2he_jaccard.data / (source_size + target_size - self.he2he_jaccard.data)

        # Only considering the traj2traj data
        for i, adj in enumerate(adjs[:-2]):
            if not i:
                adjs[i] = self.filter_traj2traj_with_leakage(adj, traj_size=filtered_traj_size, mode=1)
            else:
                adjs[i] = self.filter_traj2traj_with_leakage(adj, traj_size=None, mode=2)

        # Trajectory without checkin neighbors is not allowed!!!
        if adj_t.storage.row().unique().shape[0] != batch_size:
            diff_node = list(set(range(batch_size)) - set(adj_t.storage.row().unique().tolist()))
            raise ValueError(
                f'[NeighborSampler] Trajectory without checkin neighbors after filtering by max_time is not allowed!!\n'
                f'Those samples are sample_idx:{sample_idx[diff_node]},\n'
                f'and the corresponding query trajectories are: {n_id[diff_node]},\n'
                f'the original query trajectories are: {n_id[diff_node] - self.ci_offset}.'
            )

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (sample_idx, n_id, adjs)
        out = self.convert_batch(*out)
        return out

    def filter_traj2traj_with_leakage(self, adj, traj_size, mode=1):
        """The original traj2traj topology is in adj_t, we set the value to all ones, and
        then we substitute it with traj2traj_jaccard, and keep the data within [0, 1].

        :param adj: tuple data with traj2traj information
        :param traj_size: calculated out of this function, only take into effect when mode=1
        :param mode: 1, use self.he2he_jaccard to filter, 2, use edge_attr[:, 2] to filter
        :return:
        """
        adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size = adj

        if adj_t is None:
            return adj

        row, col, value = adj_t.coo()
        if mode == 1:
            # Add epsilon in case we delete the full-overlap traj2traj
            epsilon = 1e-6
            he2he = coo_matrix((
                np.ones(adj_t.nnz()) + epsilon,
                (row.numpy(), col.numpy())
            ))
            size_i = he2he.shape[0]
            size_j = he2he.shape[1]
            he2he = he2he - self.he2he_jaccard.tocsc()[:size_i, :size_j].tocoo()
            he2he = he2he.tocoo()

            # Valid within [0, 1]
            valid_mask = he2he.data >= 0
            he2he = SparseTensor(
                row=torch.tensor(he2he.row[valid_mask], dtype=torch.long),
                col=torch.tensor(he2he.col[valid_mask], dtype=torch.long),
                value=torch.tensor(he2he.data[valid_mask])
            )

            if adj_t.nnz() != he2he.nnz():
                raise ValueError(f"[NeighborSampler] he2he filtered size not equal.")

            # Keep intra-user and overlaped inter-user traj2traj
            inter_threshold_mask = he2he.storage.value() <= (1 - self.inter_jaccard_threshold + epsilon)
            intra_threshold_mask = he2he.storage.value() <= (1 - self.intra_jaccard_threshold + epsilon)
            inter_user_mask = (edge_type == 1) & inter_threshold_mask
            intra_user_mask = (edge_type == 0) & intra_threshold_mask
            mask = intra_user_mask | inter_user_mask
            keep_num = torch.sum(mask).item()
            if keep_num == 0:
                adj = (None, None, None, None, None, None, e_id, size)
                return adj
            else:
                # logging.info(
                #     f"[NeighborSampler] Remaining {keep_num} traj2traj[0] edges, including "
                #     f"{intra_user_mask.sum().item()} intra-user traj2traj edges and {inter_user_mask.sum().item()} "
                #     f"inter-user traj2traj edges."
                # )
                # save jaccard metric to value
                adj_t = SparseTensor(
                    row=row[mask],
                    col=col[mask],
                    value=he2he.storage.value()[mask],
                    sparse_sizes=adj_t.sparse_sizes()
                )
                edge_t = edge_t[mask] if edge_t is not None else None

                # recover similarity metric, and calculate edge_attr
                row, col, value = adj_t.coo()
                edge_attr = (1 + epsilon) - value
                source_traj_size = torch.tensor(traj_size[row]) / self.max_traj_size
                target_traj_size = torch.tensor(traj_size[col]) / self.max_traj_size
                edge_attr = torch.stack([source_traj_size, target_traj_size, edge_attr], dim=1)
        else:
            inter_threshold_mask = edge_attr[:, 2] >= self.inter_jaccard_threshold
            intra_threshold_mask = edge_attr[:, 2] >= self.intra_jaccard_threshold
            inter_user_mask = (edge_type == 1) & inter_threshold_mask
            intra_user_mask = (edge_type == 0) & intra_threshold_mask
            mask = intra_user_mask | inter_user_mask
            keep_num = torch.sum(mask).item()
            if keep_num == 0:
                adj = (None, None, None, None, None, None, e_id, size)
                return adj
            else:
                # logging.info(
                #     f"[NeighborSampler] Remaining {keep_num} traj2traj[>0] edges, including "
                #     f"{intra_user_mask.sum().item()} intra-user traj2traj edges and {inter_user_mask.sum().item()} "
                #     f"inter-user traj2traj edges."
                # )
                edge_attr = edge_attr[mask]
                adj_t = SparseTensor(
                    row=row[mask],
                    col=col[mask],
                    value=value[mask],
                    sparse_sizes=adj_t.sparse_sizes()
                )

        edge_type = edge_type[mask]
        edge_delta_t = edge_delta_t[mask]
        edge_delta_s = edge_delta_s[mask]
        e_id = e_id[mask]
        adj = (adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, e_id, size)
        return adj

    def convert_batch(self, sample_idx, n_id, adjs):
        """
        Add target label for batch data, and update target trajectory mean_time, mean_lon, mean_lat, last_lon, last_lat.

        :param sample_idx: sample_idx from label table;
        :param n_id: original index of nodes in hypergraph;
        :param adjs: Adj data of multi-hop neighbors;
        :return: Batch data
        """
        adjs_t, edge_attrs, edge_ts, edge_types, edge_delta_ts, edge_delta_ss = [], [], [], [], [], []
        y = self.y[sample_idx]

        x_target = None

        # checkin_feature 'user_id', 'poi_id', 'poi_cat', 'time', 'poi_lon', 'poi_lat'
        # trajectory_feature 'size', 'mean_lon', 'mean_lat', 'mean_time', 'start_time', 'end_time'
        i = 0
        for adj_t, edge_attr, edge_t, edge_type, edge_delta_t, edge_delta_s, _, _ in adjs:

            if adj_t is None:
                pass

            else:
                col, row, _ = adj_t.coo()
                if not i:
                    # Update filtered_ci2traj edge information and generate x feature for target trajectory (x_target)
                    source_checkin_lon_lat = self.x[n_id[row]][:, 4:6]  # [#edge, 2]
                    traj_min_time, _ = scatter_min(edge_t, col, dim=-1)  # [N, ]
                    traj_max_time, e_id = scatter_max(edge_t, col, dim=-1)
                    traj_mean_time = scatter_mean(edge_t, col, dim=-1)  # [N, ]
                    traj_last_lon_lat = source_checkin_lon_lat[e_id]  # [N, 2]
                    traj_mean_lon_lat = scatter_mean(source_checkin_lon_lat, col, dim=0)  # [N, 2]
                    traj_size = scatter_add(torch.ones_like(edge_t), col, dim=-1)  # [N, ]

                    edge_delta_t = traj_max_time[col] - edge_t
                    edge_delta_s = torch.cat([traj_last_lon_lat[col], source_checkin_lon_lat], dim=-1)
                    edge_delta_s = haversine(
                        edge_delta_s[:, 0],
                        edge_delta_s[:, 1],
                        edge_delta_s[:, 2],
                        edge_delta_s[:, 3]
                    )
                    x_target = torch.cat([
                        traj_size.unsqueeze(1),
                        traj_mean_lon_lat,
                        traj_mean_time.unsqueeze(1),
                        traj_min_time.unsqueeze(1),
                        traj_max_time.unsqueeze(1)],
                        dim=-1
                    )
                elif i == len(adjs) - 1:
                    # Update traj2traj edge information for one-hop neighbor -> target
                    edge_delta_t = x_target[col][:, 3] - self.x[n_id[row]][:, 3]
                    edge_delta_s = torch.cat([self.x[n_id[row]][:, 1:3], x_target[col][:, 1:3]], dim=-1)
                    edge_delta_s = haversine(
                        edge_delta_s[:, 0],
                        edge_delta_s[:, 1],
                        edge_delta_s[:, 2],
                        edge_delta_s[:, 3]
                    )
                else:
                    pass

            adjs_t.append(adj_t)
            edge_ts.append(edge_t)
            edge_attrs.append(edge_attr)
            edge_types.append(edge_type)
            edge_delta_ts.append(edge_delta_t)
            edge_delta_ss.append(edge_delta_s)
            i += 1

        result = Batch(
            sample_idx=sample_idx,
            x=self.x[n_id],
            x_target=x_target,
            y=y,
            adjs_t=adjs_t,
            edge_attrs=edge_attrs,
            edge_ts=edge_ts,
            edge_types=edge_types,
            edge_delta_ts=edge_delta_ts,
            edge_delta_ss=edge_delta_ss
        )
        return result

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


class Batch(NamedTuple):
    sample_idx: Tensor
    x: Tensor
    x_target: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]
    edge_attrs: List[Tensor]
    edge_ts: List[Tensor]
    edge_types: List[Tensor]
    edge_delta_ts: List[Tensor]
    edge_delta_ss: List[Tensor]

    def to(self, *args, **kwargs):
        return Batch(
            sample_idx=self.sample_idx.to(*args, **kwargs),
            x=self.x.to(*args, **kwargs),
            x_target=self.x_target.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) if adj_t is not None else None for adj_t in self.adjs_t],
            edge_attrs=[
                edge_attr.to(*args, **kwargs)
                if edge_attr is not None
                else None
                for edge_attr in self.edge_attrs
            ],
            edge_ts=[
                edge_t.to(*args, **kwargs)
                if edge_t is not None
                else None
                for edge_t in self.edge_ts
            ],
            edge_types=[
                edge_type.to(*args, **kwargs)
                if edge_type is not None
                else None
                for edge_type in self.edge_types
            ],
            edge_delta_ts=[
                edge_delta_t.to(*args, **kwargs)
                if edge_delta_t is not None
                else None
                for edge_delta_t in self.edge_delta_ts
            ],
            edge_delta_ss=[
                edge_delta_s.to(*args, **kwargs)
                if edge_delta_s is not None
                else None
                for edge_delta_s in self.edge_delta_ss
            ]
        )
