import torch
from torch import Tensor
from torch_sparse.storage import SparseStorage, get_layout
from torch_sparse.tensor import SparseTensor
from torch_sparse import spspmm, masked_select_nnz
from math import radians
from math import cos
from math import sin
from math import asin
from math import sqrt
from math import exp
import pandas as pd
from bisect import bisect
import numpy as np
import time
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def cal_slot_distance(value, slots):
    """
    Calculate a value's distance with nearest lower bound and higher bound in slots.
    :param value: The value to be calculated.
    :param slots: values of slots, needed to be sorted.
    :return: normalized distance with lower bound and higher bound,
        and index of lower bound and higher bound.
    """
    time1 = time.time()
    higher_bound = bisect(slots, value)
    time2 = time.time()
    lower_bound = higher_bound - 1
    if higher_bound == len(slots):
        return 1., 0., lower_bound, lower_bound, time2 - time1
    else:
        lower_value = slots[lower_bound]
        higher_value = slots[higher_bound]
        total_distance = higher_value - lower_value
        return (value - lower_value) / total_distance, (
            higher_value - value) / total_distance, lower_bound, higher_bound, time2 - time1


def cal_slot_distance_batch(batch_value, slots):
    """
    Proceed `cal_slot_distance` on a batch of data.
    :param batch_value: a batch of value, size (batch_size, step)
    :param slots: values of slots, needed to be sorted.
    :return: batch of distances and indexes. All with shape (batch_size, step).
    """
    # Lower bound distance, higher bound distance, lower bound, higher bound.

    ld, hd, l, h = [], [], [], []
    time_cost_list = []
    for step in batch_value:
        ld_one, hd_one, l_one, h_one, time_cost = cal_slot_distance(step, slots)
        ld.append(ld_one)
        hd.append(hd_one)
        l.append(l_one)
        h.append(h_one)
        time_cost_list.append(time_cost)
    print(f"total bisect time: {sum(time_cost_list)}")
    # return np.array(ld), np.array(hd), np.array(l), np.array(h)

    # with parallel_backend("threading", n_jobs=100):
    #     res = Parallel()(delayed(cal_slot_distance)(step, slots) for step in batch_value)
    # ld, hd, l, h = zip(*res)

    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     res = [executor.submit(cal_slot_distance, step, slots) for step in batch_value]
    #     res = [future.result() for future in as_completed(res)]
    #     ld, hd, l, h = zip(*res)
    return torch.tensor(ld), torch.tensor(hd), torch.tensor(l), torch.tensor(h)


def construct_slots(min_value, max_value, num_slots, type):
    """
    Construct values of slots given min value and max value.
    :param min_value: minimum value.
    :param max_value: maximum value.
    :param num_slots: number of slots to construct.
    :param type: type of slots to construct, 'linear' or 'exp'.
    :return: values of slots.
    """
    if type == 'exp':
        n = (max_value - min_value) / (exp(num_slots - 1) - 1)
        slots = [n * (exp(x) - 1) + min_value for x in range(num_slots)]
        slots.append(n * (num_slots - 1) + n * 100 + min_value)
        return slots
    elif type == 'linear':
        n = (max_value - min_value) / (num_slots - 1)
        slots = [n * x + min_value for x in range(num_slots-1)]
        slots.append(n*(num_slots-1)*100 + min_value)
        return slots

def unique_difference(a: Tensor, b: Tensor) -> Tensor:
    '''
    only support the difference between two tensor with unique value within
    '''
    combined = torch.cat([a,b])
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts==1]
    return difference


# coalesce and A^k, multi-hop
def multi_hop_adjacency_list(adj_t: SparseTensor, num_hop: int = 2, cs_threshold: int = 1, new_edge_type: int = 0):
    '''
    input:
        inverted ajacency list, whose values denote the edge type of the edge
    output:
        rows, cols of edge_index :: multi-hop edges in a list like [A, A^2, ..., A^{num_hop-1}, A^{num_hop}]
        edge_cs_list :: the list of values denotes the connection strength(cs) between source and target
        edge_type_list :: the raw edge_type of A, use 0 as (todo need to define new edge types for multi-hop edge)
    '''

    num_nodes = adj_t.sizes()[1]
    adj_t = adj_t.coalesce()
    edge_index = torch.stack([adj_t.storage.row().type(torch.long), adj_t.storage.col().type(torch.long)], dim=0)
    edge_cs = torch.ones_like(adj_t.storage.row(), dtype=torch.long)
    print('[UTILS] edge_index', edge_index)
    print('[UTILS] edge_cs', edge_cs)

    rows = [adj_t.storage.row()]
    cols = [adj_t.storage.col()]
    edge_cs_list = [edge_cs]
    edge_type_list = [adj_t.storage.value()]

    for i in range(num_hop):
        print('[UTILS]', i)
        edge_index, edge_value = spspmm(edge_index, edge_cs, edge_index, edge_cs, num_nodes, num_nodes, num_nodes)
        edge_cs = edge_value
        cs_mask = edge_cs >= cs_threshold
        rows += [edge_index[0][cs_mask]]
        cols += [edge_index[1][cs_mask]]
        edge_cs_list += [edge_cs[cs_mask]]
        edge_type_list += [new_edge_type * torch.ones(torch.sum(cs_mask).item(), dtype=torch.long)]
    return rows, cols, edge_cs_list, edge_type_list


# encode the multiple label feature columns
def label_encode_multiple_columns(data: Tensor, num_classes: int, encode_range: list):
    N = data.size(0)
    encode = []
    for i in encode_range:
        one_column = data[:, i].long()
        one_hot = torch.zeros(N, num_classes, device=data.device).long()
        one_hot.scatter_(dim=1, index=torch.unsqueeze(one_column, dim=1),
                src=torch.ones(N, num_classes, device=data.device).long())
        encode.append(one_hot)
    result = torch.cat(encode, dim=1)
    return result


def unique_adj_t_splits(adj_t: SparseTensor):
    """"
    Split the adj_t to two parts, one contains the src_nodes only having one target,
    the other contains the src_nodes with more than one targets
    input:
        adj_t
    output:
        unique_adj_t
        duplicate_adj_t
    """

    num_nodes = torch.max(adj_t.storage.col()).item() + 1
    src_idx, src_counts = torch.unique(adj_t.storage.col(), return_counts=True)
    unique_src = src_idx[~src_counts.ge(2)]
    #print('[DELTA] num_nodes', num_nodes)
    #print('[DELTA] unique_src', unique_src, unique_src.size())
    unique_mask = torch.tensor([False] * num_nodes)
    unique_mask[unique_src] = True
    nnz_mask = unique_mask[adj_t.storage.col()]
    duplicate_adj_t = masked_select_nnz(adj_t, mask=~nnz_mask)
    unique_adj_t = masked_select_nnz(adj_t, mask=nnz_mask)
    return unique_adj_t, duplicate_adj_t


def delta_t_calculate(x_year: Tensor, adj_t: SparseTensor):
    src_years = x_year[adj_t.storage.col()]
    tar_years = x_year[adj_t.storage.row()]
    delta_ts_pre = tar_years - src_years
    src_tar_mult = src_years * tar_years
    delta_ts = torch.where(src_tar_mult == 0, src_tar_mult, delta_ts_pre)
    return delta_ts


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    def row_wise(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return c

    if isinstance(lon1, torch.Tensor):
        if not lon1.numel():
            return None
        lon1 = torch.deg2rad(lon1)
        lat1 = torch.deg2rad(lat1)
        lon2 = torch.deg2rad(lon2)
        lat2 = torch.deg2rad(lat2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))
    elif isinstance(lon1, pd.Series):
        if not lon1.shape[0]:
            return None
        lon_lat = pd.concat([lon1, lat1, lon2, lat2], axis=1)
        c = lon_lat.apply(lambda x: row_wise(x[0], x[1], x[2], x[3]), axis=1)
    else:
        if pd.isna(lon1) or pd.isna(lat1) or pd.isna(lon2) or pd.isna(lat2):
            return None
        c = row_wise(lon1, lat1, lon2, lat2)

    r = 6371
    return c * r


if __name__ == "__main__":
    input_mode = '2'
    value = [[11.6189, -11.5651, 21.9763, 3.4623],
         [21.9763, 3.4623, -3.3726, 3.6943],
         [11.6189, -11.5651, -3.3726, 3.6943],
         [-9.7569, -7.5269, -3.3726, 3.6943],
         [-14.4530, 11.5554, -3.3726, 3.6943],
         [8.0353, 1.5913, -3.3726, 3.6943],
         [28.9380, 13.3160, -3.3726, 3.6943],
         [3.6643, -4.1383, -3.3726, 3.6943]]

    if input_mode == '1':
        tensor_input = torch.tensor(value, dtype=torch.float64)
        print(f'{type(tensor_input[:, 0])} input:')
        dis = haversine(
            tensor_input[:, 0],
            tensor_input[:, 1],
            tensor_input[:, 2],
            tensor_input[:, 3]
        )
    elif input_mode == '2':
        df_input = pd.DataFrame(value, columns=['lon1', 'lat1', 'lon2', 'lat2'])
        print(f'{type(df_input.lon1)} input:')
        dis = haversine(
            df_input.lon1,
            df_input.lat1,
            df_input.lon2,
            df_input.lat2
        )
    else:
        lon1, lat1, lon2, lat2 = value[0]
        print(f'{type(lon1)} input:')
        dis = haversine(lon1, lat1, lon2, lat2)

    print(dis)
