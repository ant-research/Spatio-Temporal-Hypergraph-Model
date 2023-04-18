import logging
import os
import os.path as osp
import datetime
import torch
import random
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
from utils import seed_torch, set_logger, Cfg, count_parameters, test_step, save_model
from layer import NeighborSampler
from dataset import LBSNDataset
from model import STHGCN, SequentialTransformer


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
    args = parser.parse_args()
    conf_file = args.yaml_file

    cfg = Cfg(conf_file)

    hparam_dict = dict(vars(cfg.run_args).items() | vars(cfg.conv_args).items() | vars(cfg.model_args).items() | vars(
        cfg.dataset_args).items() | vars(cfg.seq_transformer_args).items())
    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    cfg.model_args.sizes = sizes

    if cfg.run_args.seed is None:
        seed = random.randint(0, 1000000)
    else:
        seed = cfg.run_args.seed
    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg.run_args.save_path is None:
        save_path = f'tensorboard/{current_time}/{cfg.dataset_args.name}'
    else:
        save_path = cfg.run_args.save_path

    if cfg.run_args.log_path is None:
        log_path = f'log/{current_time}/{cfg.dataset_args.name}'
    else:
        log_path = cfg.run_args.log_path

    cfg.run_args.save_path = save_path
    cfg.run_args.log_path = log_path

    if not osp.isdir(save_path):
        os.makedirs(save_path)
    if not osp.isdir(log_path):
        os.makedirs(log_path)

    set_logger(cfg.run_args)
    summary_writer = SummaryWriter(log_dir=save_path)

    # cuda setting
    if int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device

    # Preprocess data
    preprocess(cfg)

    # Initialize dataset
    lbsn_dataset = LBSNDataset(cfg)
    cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.dataset_args.num_user = lbsn_dataset.num_user
    cfg.dataset_args.num_poi = lbsn_dataset.num_poi
    cfg.dataset_args.num_category = lbsn_dataset.num_category
    cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id

    # Initialize neighbor sampler(dataloader)
    sample_result_train = NeighborSampler(
        lbsn_dataset.x,
        lbsn_dataset.edge_index,
        lbsn_dataset.edge_attr,
        intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
        inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
        edge_t=lbsn_dataset.edge_t,
        edge_delta_t=lbsn_dataset.edge_delta_t,
        edge_type=lbsn_dataset.edge_type,
        sizes=sizes,
        sample_idx=lbsn_dataset.sample_idx_train,
        node_idx=lbsn_dataset.node_idx_train,
        edge_delta_s=lbsn_dataset.edge_delta_s,
        max_time=lbsn_dataset.max_time_train,
        label=lbsn_dataset.label_train,
        batch_size=cfg.run_args.batch_size,
        num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
        shuffle=True,
        pin_memory=True
    )

    if cfg.model_args.name == 'sthgcn':
        model = STHGCN(cfg)
    elif cfg.model_args.name == 'seq_transformer':
        model = SequentialTransformer(cfg)
    else:
        raise NotImplementedError(
            f'[Training] Model {cfg.model_args.name}, please choose from ["sthgcn", "seq_transformer"]'
        )

    model = model.to(device)
    logging.info(f'[Training] Seed: {seed}')
    logging.info('[Training] Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info(f'[Training] Parameter {name}: {param.size()}, require_grad = {param.requires_grad}')
    logging.info(f'[Training] #Parameters: {count_parameters(model)}')

    if cfg.run_args.do_train:
        current_learning_rate = cfg.run_args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        if cfg.run_args.warm_up_steps:
            warm_up_steps = cfg.run_args.warm_up_steps
        else:
            warm_up_steps = cfg.run_args.max_steps // 2

        init_step = 0
        if cfg.run_args.init_checkpoint:
            # Restore model from checkpoint directory
            # manually set in yml
            logging.info(f'[Training] Loading checkpoint %s...' % cfg.run_args.init_checkpoint)
            checkpoint = torch.load(osp.join(cfg.run_args.init_checkpoint, 'checkpoint.pt'))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            cooldown_rate = checkpoint['cooldown_rate']
            sizes = checkpoint['sizes']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info(f'[Training] Randomly Initializing Model...')
            init_step = 0
        step = init_step

        # Set valid dataloader as it would be evaluated during training
        logging.info(f'[Training] Initial learning rate: {current_learning_rate}')

        # Training Loop
        best_metrics = 0.0
        global_step = 0
        for eph in range(cfg.run_args.epoch):
            training_logs = []
            if global_step >= cfg.run_args.max_steps:
                break
            for data in tqdm(sample_result_train):
                model.train()
                split_index = torch.max(data.adjs_t[1].storage.row()).tolist()
                data = data.to(device)
                input_data = {
                    'x': data.x,
                    'edge_index': data.adjs_t,
                    'edge_attr': data.edge_attrs,
                    'split_index': split_index,
                    'delta_ts': data.edge_delta_ts,
                    'delta_ss': data.edge_delta_ss,
                    'edge_type': data.edge_types
                }

                out, loss = model(input_data, label=data.y[:, 0])
                training_logs.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                summary_writer.add_scalar(f'train/loss_step', loss, global_step)

                if cfg.run_args.do_valid and global_step % cfg.run_args.valid_steps == 0:
                    logging.info(f'[Evaluating] Evaluating on Valid Dataset...')

                    sample_result_valid = NeighborSampler(
                        lbsn_dataset.x,
                        lbsn_dataset.edge_index,
                        lbsn_dataset.edge_attr,
                        intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
                        inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
                        edge_t=lbsn_dataset.edge_t,
                        edge_delta_t=lbsn_dataset.edge_delta_t,
                        edge_type=lbsn_dataset.edge_type,
                        sizes=sizes,
                        sample_idx=lbsn_dataset.sample_idx_valid,
                        node_idx=lbsn_dataset.node_idx_valid,
                        edge_delta_s=lbsn_dataset.edge_delta_s,
                        max_time=lbsn_dataset.max_time_valid,
                        label=lbsn_dataset.label_valid,
                        batch_size=cfg.run_args.eval_batch_size,
                        num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
                        shuffle=False,
                        pin_memory=True
                    )
                    logging.info(f'[Evaluating] Epoch {eph}, step {global_step}:')
                    recall_res, ndcg_res, map_res, mrr_res, eval_loss = test_step(model, data=sample_result_valid)
                    summary_writer.add_scalar(f'validate/Recall@1', 100*recall_res[1], global_step)
                    summary_writer.add_scalar(f'validate/Recall@5', 100*recall_res[5], global_step)
                    summary_writer.add_scalar(f'validate/Recall@10', 100*recall_res[10], global_step)
                    summary_writer.add_scalar(f'validate/Recall@20', 100*recall_res[20], global_step)
                    summary_writer.add_scalar(f'validate/MRR', mrr_res, global_step)
                    summary_writer.add_scalar(f'validate/eval_loss', eval_loss, global_step)
                    summary_writer.add_scalar('train/learning_rate', current_learning_rate, global_step)

                    metrics = 4 * recall_res[1] + recall_res[20]

                    # save model based on compositional recall metrics
                    if metrics > best_metrics:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'cooldown_rate': cfg.run_args.cooldown_rate,
                            'sizes': sizes
                        }
                        logging.info(f'[Training] Save model at step {global_step} epoch {eph}')
                        save_model(model, optimizer, save_variable_list, cfg.run_args, hparam_dict)
                        best_metrics = metrics

                # learning rate schedule
                if global_step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info(f'[Training] Change learning_rate to {current_learning_rate} at step {global_step}')
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * cfg.run_args.cooldown_rate

                if global_step >= cfg.run_args.max_steps:
                    break
                global_step += 1

            epoch_loss = sum([loss for loss in training_logs]) / len(training_logs)
            logging.info(f'[Training] Average train loss at step {global_step} is {epoch_loss}:')
            summary_writer.add_scalar('train/loss_epoch', epoch_loss, eph)

    if cfg.run_args.do_test:
        logging.info('[Evaluating] Start evaluating on test set...')
        sample_result_test = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_test,
            node_idx=lbsn_dataset.node_idx_test,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_test,
            label=lbsn_dataset.label_test,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        checkpoint = torch.load(osp.join(cfg.run_args.save_path, 'checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        recall_res, ndcg_res, map_res, mrr_res, loss = test_step(model, sample_result_test)
        num_params = count_parameters(model)
        metric_dict = {
            'hparam/num_params': num_params,
            'hparam/Recall@1': recall_res[1],
            'hparam/Recall@5': recall_res[5],
            'hparam/Recall@10': recall_res[10],
            'hparam/Recall@20': recall_res[20],
            'hparam/NDCG@1': ndcg_res[1],
            'hparam/NDCG@5': ndcg_res[5],
            'hparam/NDCG@10': ndcg_res[10],
            'hparam/NDCG@20': ndcg_res[20],
            'hparam/MAP@1': map_res[1],
            'hparam/MAP@5': map_res[5],
            'hparam/MAP@10': map_res[10],
            'hparam/MAP@20': map_res[20],
            'hparam/MRR': mrr_res,
        }
        logging.info(f'[Evaluating] Test evaluation result : {metric_dict}')
        summary_writer.add_hparams(hparam_dict, metric_dict)
        summary_writer.close()
