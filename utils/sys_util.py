import os
import random
import logging
import torch
import numpy as np
import os.path as osp


def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("Spatio-Temporal-Hypergraph-Model")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def set_logger(args):
    """
    Write logs to checkpoint and console
    """
    if args.do_train:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'train.log')
    else:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'test.log')

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w+'
    )


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
