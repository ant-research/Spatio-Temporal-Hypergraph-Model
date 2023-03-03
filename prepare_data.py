from dataset import LBSNDataset
import logging
import os
import os.path as osp
import json
import datetime
from tqdm import tqdm
from preprocess import TsmcFileReader
from generate_hypergraph import generate_hypergraph_from_file
from conf_parser import Cfg


if __name__ == '__main__':
    cfg = Cfg("run.yml")
    # Preprocess
    data_dir = osp.join('data', cfg.run_args.dataset)
    if cfg.run_args.dataset == 'nyc':
        raw_file = 'dataset_TSMC2014_NYC.txt'
    elif cfg.run_args.dataset == 'tky':
        raw_file = 'dataset_TSMC2014_TKY.txt'
    elif cfg.run_args.dataset == 'gowalla':
        raw_file = 'dataset_gowalla_ca_ne.csv'
    else:
        raise ValueError(f'Wrong dataset name: {cfg.run_args.dataset} ')

    # todo whether to overwrite old data
    processed_path = osp.join(data_dir, 'processed')
    if osp.exists(processed_path):
        TsmcFileReader.root_path = data_dir
        data = TsmcFileReader.read_dataset(raw_file, cfg.run_args.dataset)
        data = TsmcFileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
        data = TsmcFileReader.split_train_test(data, cfg.dataset_args.train_test_split_mode)
        if cfg.run_args.dataset == 'gowalla':
            data = TsmcFileReader.do_filter(data, cfg.dataset_args.min_poi_freq, cfg.dataset_args.min_user_freq)
            data = TsmcFileReader.split_train_test(data, cfg.dataset_args.train_test_split_mode)
        data = TsmcFileReader.generate_id(
            data,
            cfg.dataset_args.session_time_interval,
            cfg.dataset_args.aoi_dis_interval,
            cfg.dataset_args.do_label_encoder,
            cfg.dataset_args.only_last_metric
        )
        data.to_csv(osp.join(TsmcFileReader.root_path, 'processed', 'sample.txt'), index=False)
    else:
        os.makedirs(processed_path)

    # Hypergraph generate
    processed_file = osp.join(processed_path, 'sample.txt')
    generate_hypergraph_from_file(processed_file, processed_path, cfg.dataset_args)
