import logging
import torch
import pandas as pd
import os.path as osp
from utils import get_root_dir, construct_slots


class LBSNDataset:
    def __init__(self, cfg):
        self.data_path = osp.join(get_root_dir(), 'data', cfg.dataset_args.dataset_name, 'preprocessed')
        self.padding_poi_id, self.padding_poi_category, self.padding_user_id = 0, 0, 0
        self.padding_hour_id, self.padding_weekday_id = 0, 0
        self.num_user, self.num_poi, self.num_category, self.num_checkin, self.num_traj = 0, 0, 0, 0, 0

        df, df_train, df_valid, df_test, ci2traj, traj2traj = self.read()
        self.x = [traj2traj.x, ci2traj.x]
        self.edge_index = [traj2traj.edge_index, ci2traj.edge_index]
        self.edge_attr = [traj2traj.edge_attr, None]
        self.edge_t = [None, ci2traj.edge_t]
        self.edge_delta_t = [traj2traj.edge_delta_t, ci2traj.edge_delta_t]
        self.edge_delta_s = [traj2traj.edge_delta_s, ci2traj.edge_delta_s]
        self.edge_type = [traj2traj.edge_type, None]

        self.checkin_offset = torch.as_tensor([df.check_ins_id.max() + 1], dtype=torch.long)
        self.node_idx_train = self.get_node_id(df_train)
        self.node_idx_valid = self.get_node_id(df_valid)
        self.node_idx_test = self.get_node_id(df_test)
        self.max_time_train = self.get_max_time(df_train)
        self.max_time_valid = self.get_max_time(df_valid)
        self.max_time_test = self.get_max_time(df_test)
        self.label_train = self.get_label(df_train)
        self.label_valid = self.get_label(df_valid)
        self.label_test = self.get_label(df_test)
        self.sample_idx_train = self.get_sample_id(df_train)
        self.sample_idx_valid = self.get_sample_id(df_valid)
        self.sample_idx_test = self.get_sample_id(df_test)

        self.min_d, self.max_d = 1e8, 0.
        delta_s = torch.cat([ci2traj.edge_delta_s, traj2traj.edge_delta_s], dim=0)

        self.min_d = min(self.min_d, delta_s.min())
        self.max_d_chj2traj = max(self.max_d, ci2traj.edge_delta_s.max())
        self.max_d_tj2traj = max(self.max_d, traj2traj.edge_delta_s.max())
        self.max_d_tj2traj += cfg.dataset_args.max_d_epsilon

        if cfg.model_args.distance_encoder_type == 'hstlstm':
            self.spatial_slots = construct_slots(
                self.min_d,
                self.max_d,
                cfg.dataset_args.num_spatial_slots,
                cfg.dataset_args.spatial_slot_type
            )
        else:
            self.spatial_slots = self.min_d, self.max_d_chj2traj, self.max_d_tj2traj

        logging.info(f'[Initialize Dataset] #user: {self.num_user}')
        logging.info(f'[Initialize Dataset] #poi: {self.num_poi}')
        logging.info(f'[Initialize Dataset] #category: {self.num_category}')
        logging.info(f'[Initialize Dataset] #checkin: {self.num_checkin}')
        logging.info(f'[Initialize Dataset] #trajectory: {self.num_traj}')
        logging.info(f'[Initialize Dataset] #training_sample: {self.sample_idx_train.shape[0]}')
        logging.info(f'[Initialize Dataset] #validation_sample: {self.sample_idx_valid.shape[0]}')
        logging.info(f'[Initialize Dataset] #testing_sample: {self.sample_idx_test.shape[0]}')

    def read(self):
        df = pd.read_csv(osp.join(self.data_path, 'sample.csv')).reset_index(drop=True)
        le_data = pd.read_pickle(osp.join(self.data_path, 'label_encoding.pkl'))
        self.padding_poi_id = le_data[5]
        self.padding_poi_category = le_data[6]
        self.padding_user_id = le_data[7]
        self.padding_hour_id = le_data[8]
        self.padding_weekday_id = le_data[9]

        self.num_user = df['UserId'].nunique()
        self.num_poi = df['PoiId'].nunique()
        self.num_category = df['PoiCategoryId'].nunique()
        self.num_checkin = df.shape[0]
        self.num_traj = df['pseudo_session_trajectory_id'].nunique()

        df_train = pd.read_csv(osp.join(self.data_path, 'train_sample.csv'), sep=',')
        df_valid = pd.read_csv(osp.join(self.data_path, 'validate_sample.csv'), sep=',')
        df_test = pd.read_csv(osp.join(self.data_path, 'test_sample.csv'), sep=',')

        ci2traj = torch.load(osp.join(self.data_path, 'ci2traj_pyg_data.pt'))
        traj2traj = torch.load(osp.join(self.data_path, 'traj2traj_pyg_data.pt'))

        return df, df_train, df_valid, df_test, ci2traj, traj2traj

    def get_node_id(self, df):
        query_id = torch.tensor(df.query_pseudo_session_trajectory_id, dtype=torch.long)
        node_id = query_id + self.checkin_offset
        return node_id

    @staticmethod
    def get_max_time(df):
        max_time = torch.tensor(df.last_checkin_epoch_time, dtype=torch.long)
        return max_time

    @staticmethod
    def get_label(df):
        poi_id = torch.tensor(df.PoiId, dtype=torch.long)
        cate_id = torch.tensor(df.PoiCategoryId, dtype=torch.long)
        longitude = torch.tensor(df.Longitude, dtype=torch.float)
        latitude = torch.tensor(df.Latitude, dtype=torch.float)
        time_hour = torch.tensor(pd.to_datetime(df['UTCTimeOffset']).dt.hour / 24, dtype=torch.float)
        y = torch.stack([poi_id, cate_id, longitude, latitude, time_hour], dim=-1)
        return y

    @staticmethod
    def get_sample_id(df):
        sample_id = torch.tensor(df.index, dtype=torch.long)
        return sample_id
