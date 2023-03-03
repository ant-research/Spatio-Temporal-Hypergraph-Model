import yaml
import pathlib


class DictToObject(object):
    def __init__(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        return str(self.__dict__)


class Cfg:
    def __init__(self, file_name):
        with open(pathlib.Path.cwd() / 'conf' / file_name, "r") as f:
            conf = yaml.safe_load(f)
            self.model_args = DictToObject(conf['model_args'])
            self.delta_args = DictToObject(conf['delta_args'])
            self.transformer_args = DictToObject(conf['transformer_args'])
            self.run_args = DictToObject(conf['run_args'])
            self.dataset_args = DictToObject(conf['dataset_args'])


if __name__ == "__main__":
    cfg = Cfg("run.yml")
    print(cfg.run_args)
    print(cfg.delta_args)
