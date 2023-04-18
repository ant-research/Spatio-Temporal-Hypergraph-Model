from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from utils import haversine
import os
import os.path as osp
import logging


def generate_hypergraph_from_file(input_file, output_path, args):
    """
    Construct incidence matrix of [Checkin -> Trajectory] and adjcency list of [Trajectory -> Trajectory]
    from the raw record, the edge_index will be like
        [[ -CheckIn- ]
         [ -Trajectory(hyperedge)]
    and
        [[ -Trajectory(hyperedge)- ]
         [ -Trajectory(hyperedge)]
    separately.

    Use columns in txt file for next-poi task:
        UserId, check_ins_id, PoiId, Latitude, Longitude, PoiCategoryId, UTCTimeOffsetEpoch,
        pseudo_session_trajectory_id, UTCTimeOffsetWeekday, UTCTimeOffsetHour.

    The two part will save as two .pt files.

    :param input_file: the hypergraph raw path
    :param output_path: pyg_data.pt output directory
    :param args: parameters parsed for input
    :return: None
    """
    usecols = [
        'UserId', 'PoiId', 'PoiCategoryId', 'Latitude', 'Longitude', 'UTCTimeOffsetEpoch', 'UTCTimeOffsetWeekday',
        'UTCTimeOffsetHour', 'check_ins_id', 'pseudo_session_trajectory_id'
    ]
    threshold = args.threshold
    filter_mode = args.filter_mode
    data = pd.read_csv(input_file, usecols=usecols)

    traj_column = 'pseudo_session_trajectory_id'

    # If True, Shift traj_id with offset #check_ins_id before saving to pyg data, which means idx of checkin are
    # in range [0, #checkin_id - 1], and idx of trajectory are in range [#checkin, #trajectory+#checkin-1]
    traj_offset = True
    if traj_offset:
        checkin_offset = torch.as_tensor([data.check_ins_id.max() + 1], dtype=torch.long)
    else:
        checkin_offset = torch.as_tensor([0], dtype=torch.long)

    traj_stat = generate_hyperedge_stat(data, traj_column)
    ci2traj_pyg_data = generate_ci2traj_pyg_data(data, traj_stat, traj_column, checkin_offset)

    traj2traj_intra_u_data = generate_traj2traj_data(
        data,
        traj_stat,
        traj_column,
        threshold,
        filter_mode=filter_mode,
        relation_type='intra'
    )
    traj2traj_inter_u_data = generate_traj2traj_data(
        data,
        traj_stat,
        traj_column,
        threshold,
        filter_mode=filter_mode,
        relation_type='inter'
    )
    traj2traj_pyg_data = merge_traj2traj_data(traj_stat, traj2traj_intra_u_data, traj2traj_inter_u_data, checkin_offset)

    # save pyg data
    if not osp.isdir(output_path):
        os.makedirs(output_path)
    ci2traj_out_file = osp.join(output_path, 'ci2traj_pyg_data.pt')
    traj2traj_out_file = osp.join(output_path, 'traj2traj_pyg_data.pt')
    torch.save(ci2traj_pyg_data, ci2traj_out_file)
    torch.save(traj2traj_pyg_data, traj2traj_out_file)

    logging.info(
        f'[Preprocess - Generate Hypergraph] Done saving checkin2trajectory pyg data to {ci2traj_out_file}'
        f' and trajectory2trajectory pyg data to {traj2traj_out_file}.'
    )
    return


def generate_hyperedge_stat(data, traj_column):
    """
    Generate trajectory hyperedge statistics data (pd.DataFrame)

    :param data: raw pseudo-session trajectory data
    :param traj_column: trajectory column name
    :return:
    """
    traj_stat = pd.DataFrame()
    traj_stat['size'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(len)
    traj_stat['mean_lon'] = data.groupby(traj_column)['Longitude'].apply(sum) / traj_stat['size']
    traj_stat['mean_lat'] = data.groupby(traj_column)['Latitude'].apply(sum) / traj_stat['size']
    traj_stat[['last_lon', 'last_lat']] = \
        data.sort_values([traj_column, 'UTCTimeOffsetEpoch']).groupby(traj_column).last()[['Longitude', 'Latitude']]

    traj_stat['start_time'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(min)
    traj_stat['end_time'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(max)
    traj_stat['mean_time'] = data.groupby(traj_column)['UTCTimeOffsetEpoch'].apply(sum) / traj_stat['size']
    traj_stat['time_window_in_hour'] = (traj_stat.end_time - traj_stat.start_time) / (60*60)
    logging.info(f'[Preprocess - Generate Hypergraph] Number of hyperedges(trajectory): {traj_stat.shape[0]}.')
    logging.info(
        f'[Preprocess - Generate Hypergraph] The min, mean, max size of hyperedges are: '
        f'{traj_stat["size"].min()}, {traj_stat["size"].mean()}, {traj_stat["size"].max()}.'
    )
    logging.info(
        f'[Preprocess - Generate Hypergraph] The min, mean, max time window of hyperedges are:'
        f'{traj_stat.time_window_in_hour.min()}, {traj_stat.time_window_in_hour.mean()}, '
        f'{traj_stat.time_window_in_hour.max()}.'
    )
    return traj_stat


def generate_ci2traj_pyg_data(data, traj_stat, traj_column, checkin_offset):
    """
    Generate checkin2trajectory incidence matrix, checkin (here ci is short for checkin) feature matrix, and
    edge_delta_t and edge_delta_s. Then store them into pyg data.
    edge_delta_t is calculated by (traj(max_time) - current_time)
    edge_delta_s is calculated by (geodis(traj(last_lbs), current_lbs))

    :param data: raw trajectory data;
    :param traj_stat: hyperedge(trajectory) statistics;
    :param traj_column: trajectory column name;
    :param checkin_offset: max checkin index plus 1;
    :return: pyg_data including incidence matrix and checkin feature matrix and other edge information.
    """
    checkin_feature_columns = [
        'UserId',
        'PoiId',
        'PoiCategoryId',
        'UTCTimeOffsetEpoch',
        'Longitude',
        'Latitude',
        'UTCTimeOffsetWeekday',
        'UTCTimeOffsetHour'
    ]
    checkin_feature = data.sort_values('check_ins_id')[checkin_feature_columns].to_numpy()
    assert data.check_ins_id.unique().shape[0] == data.check_ins_id.max() + 1, \
        'check_ins_id is not chronological order in raw data'

    # Calculate distance between trajectory's last poi location and curren poi location
    delta_s_in_traj = data.join(traj_stat, on=traj_column, how='left')[
        ['Longitude', 'Latitude', 'last_lon', 'last_lat']
    ]
    delta_s_in_traj['distance_km'] = haversine(
        delta_s_in_traj.Longitude,
        delta_s_in_traj.Latitude,
        delta_s_in_traj.last_lon,
        delta_s_in_traj.last_lat
    )

    # Create incidence matrix for check-in -> trajectory
    ci2traj_adj_t = SparseTensor(
        row=torch.as_tensor(data[traj_column].tolist(), dtype=torch.long),
        col=torch.as_tensor(data.check_ins_id.tolist(), dtype=torch.long),
        value=torch.as_tensor(range(0, data.shape[0]), dtype=torch.long)
    )
    perm = ci2traj_adj_t.storage.value()
    ci2traj_edge_t = torch.tensor(data.UTCTimeOffsetEpoch.tolist())[perm]
    ci2traj_edge_delta_t = torch.tensor(
        traj_stat.end_time[data[traj_column].tolist()].values - data.UTCTimeOffsetEpoch.values
    )[perm]
    ci2traj_edge_delta_s = torch.tensor(delta_s_in_traj.distance_km.tolist())[perm]

    ci2traj_edge_index = torch.stack([ci2traj_adj_t.storage.col(), ci2traj_adj_t.storage.row() + checkin_offset])

    ci2traj_pyg_data = Data(
        edge_index=ci2traj_edge_index,
        x=torch.tensor(checkin_feature),
        edge_t=ci2traj_edge_t,
        edge_delta_t=ci2traj_edge_delta_t,
        edge_delta_s=ci2traj_edge_delta_s
    )
    ci2traj_pyg_data.num_hyperedges = traj_stat.shape[0]
    return ci2traj_pyg_data


def generate_traj2traj_data(
        data,
        traj_stat,
        traj_column,
        threshold=0.02,
        filter_mode='min size',
        chunk_num=10,
        relation_type='intra'
):
    """
    Generate hyperedge2hyperedge (traj2traj) dynamic relation.

    :param data: raw trajectory data;
    :param traj_stat: hyperedge(trajectory) statistics;
    :param traj_column: trajectory column name;
    :param threshold: threshold for filtering noise relation;
    :param filter_mode: filter mode for filtering noise relation;
    :param chunk_num: number of chunk for fast filtering.
    :param relation_type: intra or inter, switch for different type of hyperedge2hyperedge relation.
    :return: hyperedge2hyperedge tuple data(edge_index(coo), edge_type, edge_delta_t and edge_delta_s.
    """
    traj2traj_original_metric = None
    # First create sparse matrix for trajectory -> poi, then generate inter-user adjacency list
    # one trajectory may have multiple identical poi_id, we drop the duplicate ones first
    traj_user_map = data[['UserId', traj_column]].drop_duplicates().set_index(traj_column)
    traj_size_adjust = None
    if relation_type == 'inter':
        traj_poi_map = data[['PoiId', traj_column]].drop_duplicates()
        traj2node = coo_matrix((
            np.ones(traj_poi_map.shape[0]),
            (np.array(traj_poi_map['PoiId'], dtype=np.int64), np.array(traj_poi_map[traj_column], dtype=np.int64))
        )).tocsr()

        # adjust the traj_id size based on new traj_poi_map
        traj_size_adjust = traj_poi_map.groupby(traj_column).apply(len).tolist()
    else:
        traj2node = coo_matrix((
            np.ones(traj_user_map.shape[0]),
            (np.array(traj_user_map['UserId'], dtype=np.int64), np.array(traj_user_map.index, dtype=np.int64))
        )).tocsr()

    node2traj = traj2node.T
    traj2traj = node2traj * traj2node
    traj2traj = traj2traj.tocoo()

    # for inter-user type, save the original similarity metric
    if relation_type == 'inter':
        row_filtered, col_filtered, data_filtered = filter_chunk(
            row=traj2traj.row,
            col=traj2traj.col,
            data=traj2traj.data,
            chunk_num=chunk_num,
            he_size=traj_size_adjust,
            threshold=0,
            filter_mode=filter_mode
        )
        traj2traj_original_metric = coo_matrix((data_filtered, (row_filtered, col_filtered)), shape=traj2traj.shape)

    # Filter 1: filter based on pre-define conditions
    # 1. different trajectory 2. source_endtime <= target_starttime
    mask_1 = traj2traj.row != traj2traj.col
    mask_2 = traj_stat.end_time[traj2traj.col].values <= traj_stat.start_time[traj2traj.row].values
    mask = mask_1 & mask_2
    if relation_type == 'inter':
        # 3. diffrent user
        mask_3 = traj_user_map['UserId'][traj2traj.row].values != traj_user_map['UserId'][traj2traj.col].values
        mask = mask & mask_3

    traj2traj.row = traj2traj.row[mask]
    traj2traj.col = traj2traj.col[mask]
    traj2traj.data = traj2traj.data[mask]

    if relation_type == 'inter':
        # Filter 2: filter based on pre-define metric threshold
        row_filtered, col_filtered, data_filtered = filter_chunk(
            row=traj2traj.row,
            col=traj2traj.col,
            data=traj2traj.data,
            chunk_num=chunk_num,
            he_size=traj_size_adjust,
            threshold=threshold,
            filter_mode=filter_mode
        )
        traj2traj.row = row_filtered
        traj2traj.col = col_filtered
        traj2traj.data = data_filtered
        edge_type = np.ones_like(traj2traj.row)
    else:
        edge_type = np.zeros_like(traj2traj.row)

    # Calculate edge_delta_t and edge_delta_s
    edge_delta_t = traj_stat.mean_time[traj2traj.row].values - traj_stat.mean_time[traj2traj.col].values
    edge_delta_s = np.stack([
        traj_stat.mean_lon[traj2traj.row].values,
        traj_stat.mean_lat[traj2traj.row].values,
        traj_stat.mean_lon[traj2traj.col].values,
        traj_stat.mean_lat[traj2traj.col].values],
        axis=1
    )

    edge_delta_s = torch.tensor(edge_delta_s)
    edge_delta_s = haversine(edge_delta_s[:, 0], edge_delta_s[:, 1], edge_delta_s[:, 2], edge_delta_s[:, 3])

    logging.info(
        f"[Preprocess - Generate Hypergraph] Number of {relation_type}-user hyperedge2hyperedge(traj2traj) "
        f"relation has been generated: {traj2traj.row.shape[0]}, while threshold={threshold} and mode={filter_mode}."
    )

    return traj2traj, traj2traj_original_metric, edge_type, edge_delta_t, edge_delta_s.numpy()


def merge_traj2traj_data(traj_stat, intra_u_data, inter_u_data, checkin_offset):
    """
    Merge intra-user and inter-user hyperedge2hyperedge(traj2traj) dynamic relation.
    Merge intra-user and inter-user hyperedge2hyperedge(traj2traj) dynamic relation.

    :param traj_stat: hyperedge(trajectory) statistics;
    :param intra_u_data: hyperedge2hyperedge(traj2traj) relation between the same user, composited of tuple with
        edge_index(coo), edge_attr(np.array), edge_type(np.array), edge_delta_t(np.array), edge_delta_s(np.array);
    :param inter_u_data: hyperedge2hyperedge(traj2traj) relation between different users, composited of tuple like
        intra_u_data.
    :param checkin_offset: max checkin index plus 1;
    :return: pyg data of traj2traj
    """
    traj_feature = traj_stat[['size', 'mean_lon', 'mean_lat', 'mean_time', 'start_time', 'end_time']].to_numpy()

    # add two extra feature column to make sure traj feature has the same dimension size with ci feature
    padding_feature = np.zeros([traj_feature.shape[0], 2])
    traj_feature = np.concatenate([traj_feature, padding_feature], axis=1)

    intra_edge_index, _, intra_edge_type, intra_edge_delta_t, intra_edge_delta_s = intra_u_data
    inter_edge_index, traj2traj_orginal_metric, inter_edge_type, inter_edge_delta_t, inter_edge_delta_s = inter_u_data
    row = np.concatenate([intra_edge_index.row, inter_edge_index.row])
    col = np.concatenate([intra_edge_index.col, inter_edge_index.col])

    # replace data with metric value
    metric_data = coo_matrix((np.ones(row.shape[0]), (row, col)), shape=traj2traj_orginal_metric.shape)
    epsilon = coo_matrix((np.zeros(row.shape[0]) + 1e-6, (row, col)), shape=traj2traj_orginal_metric.shape)
    metric_data = metric_data.multiply(traj2traj_orginal_metric)
    metric_data += epsilon

    adj_t = SparseTensor(
        row=torch.as_tensor(row, dtype=torch.long),
        col=torch.as_tensor(col, dtype=torch.long),
        value=torch.as_tensor(range(0, row.shape[0]), dtype=torch.long)
    )
    perm = adj_t.storage.value()

    x = torch.tensor(traj_feature)
    edge_type = torch.tensor(np.concatenate([intra_edge_type, inter_edge_type]))[perm]
    edge_delta_t = torch.tensor(np.concatenate([intra_edge_delta_t, inter_edge_delta_t]))[perm]
    edge_delta_s = torch.tensor(np.concatenate([intra_edge_delta_s, inter_edge_delta_s]))[perm]

    edge_index = torch.stack([
        adj_t.storage.col() + checkin_offset,
        adj_t.storage.row() + checkin_offset
    ])

    # edge_attr: source_size, target_size, jaccard_similarity
    source_size = x[edge_index[0] - checkin_offset][:, 0] / x[:, 0].max()
    target_size = x[edge_index[1] - checkin_offset][:, 0] / x[:, 0].max()
    edge_attr = torch.stack([source_size, target_size, torch.tensor(metric_data.data)], dim=1)

    traj2traj_pyg_data = Data(
        edge_index=edge_index,
        x=x,
        edge_attr=edge_attr,
        edge_type=edge_type,
        edge_delta_t=edge_delta_t,
        edge_delta_s=edge_delta_s
    )
    return traj2traj_pyg_data


def filter_chunk(row, col, data, he_size, chunk_num=10, threshold=0.02, filter_mode='min size'):
    """
    Filter noise hyperedge2hyperedge connection based on metric threshold

    :param row: row, hyperedge2hyperedge scipy.sparse coo matrix
    :param col: col, hyperedge2hyperedge scipy.sparse coo matrix
    :param data: data, hyperedge2hyperedge scipy.sparse coo matrix
    :param he_size: hyperedge size list (drop duplicates)
    :param chunk_num: number of chunk to prevent from oom issue
    :param threshold: metric threshold, relation will be kept only if metric value is greater than threshold
    :param filter_mode: min_size - propotional to minmum size, 'jaccard' - jaccard similarity
        min_size, E2E_{ij} keeps when E2E_{ij} \ge \theta\min(|\mathcal{E}_i|,|\mathcal{E}_j|)
        jaccard, E2E_{ij} keeps when \frac{E2E_{ij}}{|\mathcal{E}_i|+|\mathcal{E}_j| - E2E_{ij}} \ge \theta
    :return:
    """
    # Split the data to multiple chunks for large data
    chunk_bin = np.linspace(0, row.shape[0], chunk_num, dtype=np.int64)
    rows, cols, datas = [], [], []
    for i in tqdm(range(len(chunk_bin) - 1)):
        row_chunk = row[chunk_bin[i]:chunk_bin[i + 1]]
        col_chunk = col[chunk_bin[i]:chunk_bin[i + 1]]
        data_chunk = data[chunk_bin[i]:chunk_bin[i + 1]]
        source_size = np.array(list(map(he_size.__getitem__, row_chunk.tolist())))
        target_size = np.array(list(map(he_size.__getitem__, col_chunk.tolist())))
        if filter_mode == 'min size':
            # propotional to minimum size
            metric = data_chunk / np.minimum(source_size, target_size)
        else:
            # jaccard similarity
            metric = data_chunk / (source_size + target_size - data_chunk)
        filter_mask = metric >= threshold
        rows.append(row_chunk[filter_mask])
        cols.append(col_chunk[filter_mask])
        datas.append(metric[filter_mask])

    return np.concatenate(rows), np.concatenate(cols), np.concatenate(datas)
