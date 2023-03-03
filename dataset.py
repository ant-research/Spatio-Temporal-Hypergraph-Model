import numpy as np
import torch
import os.path as osp
from sampler import NeighborSampler
from utils import construct_slots
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import pickle


class LBSNDataset():
    def __init__(self, data_dir, args):
        self.data_dir = data_dir
        df, df_train, df_valid, df_test, self.num_users, self.num_pois, self.num_category, self.n_session_traj, ci2traj, traj2traj, self.padding_poi_id, self.padding_poi_category, self.padding_user_id, self.padding_hour_id, self.padding_weekday_id = self.read(
            data_dir)

        self.x = [traj2traj.x, ci2traj.x]
        self.edge_index = [traj2traj.edge_index, ci2traj.edge_index]
        self.edge_attr = [traj2traj.edge_attr, None]
        self.edge_t = [None, ci2traj.edge_t]
        self.edge_delta_t = [traj2traj.edge_delta_t, ci2traj.edge_delta_t]
        self.edge_delta_s = [traj2traj.edge_delta_s, ci2traj.edge_delta_s]
        self.edge_type = [traj2traj.edge_type, None]

        checkin_offset = torch.as_tensor([df.check_ins_id.max() + 1], dtype=torch.long)

        self.node_idx_train = self.get_node_id(df_train, checkin_offset)
        self.node_idx_valid = self.get_node_id(df_valid, checkin_offset)
        self.node_idx_test = self.get_node_id(df_test, checkin_offset)

        self.max_time_train = self.get_max_time(df_train)
        self.max_time_valid = self.get_max_time(df_valid)
        self.max_time_test = self.get_max_time(df_test)

        self.label_train = self.get_label(df_train)
        self.label_valid = self.get_label(df_valid)
        self.label_test = self.get_label(df_test)

        self.class_weight = compute_class_weight(
            class_weight='balanced', classes=np.unique(df['PoiId'].tolist()), y=df['PoiId'].tolist())
        self.class_weight = torch.tensor(self.class_weight, dtype=torch.float)

        self.sample_idx_train = self.get_sample_id(df_train)
        self.sample_idx_valid = self.get_sample_id(df_valid)
        self.sample_idx_test = self.get_sample_id(df_test)

        self.min_d, self.max_d = 1e8, 0.
        # self.edge_delta_s_traj2traj = torch.tensor([1.,2,3])
        delta_s = torch.cat([ci2traj.edge_delta_s, traj2traj.edge_delta_s], dim=0)

        self.min_d = min(self.min_d, delta_s.min())
        # self.max_d = max(self.max_d, delta_s.max())
        self.max_d_chj2traj = max(self.max_d, ci2traj.edge_delta_s.max())
        self.max_d_tj2traj = max(self.max_d, traj2traj.edge_delta_s.max())
        # print(self.max_d_chj2traj)    # tensor(43.7633)
        self.max_d_tj2traj += args.max_d_epsilon

        # self.spatial_slots = construct_slots(self.min_d, self.max_d,
        #                                      args.num_spatial_slots, args.spatial_slot_type)
        self.spatial_slots = self.min_d, self.max_d_chj2traj, self.max_d_tj2traj

    def read(self, data_dir):
        df = pd.read_csv(data_dir + '/' + 'sample.txt', sep=',').reset_index(drop=True)
        _, _, _, _, _, padding_poi_id, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id = pd.read_pickle(
            data_dir + '/' + 'label_encoding.pkl')
        n_user = df['UserId'].nunique()
        n_poi = df['PoiId'].nunique()
        n_category = df['PoiCategoryId'].nunique()
        n_session_traj = df['pseudo_session_trajectory_id'].nunique()

        df_train = pd.read_csv(data_dir + '/train_sample.csv', sep=',')
        df_valid = pd.read_csv(data_dir + '/validation_sample.csv', sep=',')
        df_test = pd.read_csv(data_dir + '/test_sample.csv', sep=',')

        if osp.exists(osp.join(data_dir, 'test_traj_id.pkl')):
            print('-'*10, 'do test trajectory id filter', '-'*10)
            with open(osp.join(data_dir, 'test_traj_id.pkl'), 'rb') as f:
                traj_list = pickle.load(f)
            df_test = df_test[df_test['trajectory_id'].isin(set(traj_list))]

        ci2traj = torch.load(data_dir + '/ci2traj_pyg_data.pt')
        traj2traj = torch.load(data_dir + '/traj2traj_pyg_data.pt')

        # 去除train里面没有出现POI和user
        train_user_set = set(df_train['UserId'])
        train_poi_set = set(df_train['PoiId'])
        df_valid = df_valid[
            (df_valid['UserId'].isin(train_user_set)) & (df_valid['PoiId'].isin(train_poi_set))].reset_index()
        df_test = df_test[(df_test['UserId'].isin(train_user_set)) & (df_test['PoiId'].isin(train_poi_set))].reset_index()

        return df, df_train, df_valid, df_test, n_user, n_poi, n_category, n_session_traj, ci2traj, traj2traj, padding_poi_id, padding_poi_category, padding_user_id, padding_hour_id, padding_weekday_id

    def get_node_id(self, df, checkin_offset):
        query_id = torch.tensor(df.query_pseudo_session_trajectory_id, dtype=torch.long)
        node_id = query_id + checkin_offset
        return node_id

    def get_max_time(self, df):
        max_time = torch.tensor(df.last_checkin_epoch_time, dtype=torch.long)
        return max_time

    def get_label(self, df):
        poi_id = torch.tensor(df.PoiId, dtype=torch.long)
        cate_id = torch.tensor(df.PoiCategoryId, dtype=torch.long)
        longitude = torch.tensor(df.Longitude, dtype=torch.float)
        latitude = torch.tensor(df.Latitude, dtype=torch.float)
        time_hour = torch.tensor(pd.to_datetime(df['UTCTimeOffset']).dt.hour / 24, dtype=torch.float)
        y = torch.stack([poi_id, cate_id, longitude, latitude, time_hour], dim=-1)
        return y

    def get_sample_id(self, df):
        sample_id = torch.tensor(df.index, dtype=torch.long)
        return sample_id
