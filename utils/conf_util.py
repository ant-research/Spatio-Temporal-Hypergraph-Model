import yaml
import os.path as osp
from utils import get_root_dir


class DictToObject(object):
    def __init__(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        return str(self.__dict__)


class Cfg:
    def __init__(self, file_name):
        file_path = osp.join(get_root_dir(), 'conf', file_name)
        with open(file_path, "r") as f:
            conf = yaml.safe_load(f)
            self.model_args = DictToObject(conf.get('model_args', {}))
            self.conv_args = DictToObject(conf.get('conv_args', {}))
            self.seq_transformer_args = DictToObject(conf.get('seq_transformer_args', {}))
            self.run_args = DictToObject(conf.get('run_args', {}))
            self.dataset_args = DictToObject(conf.get('dataset_args', {}))
