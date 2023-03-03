import logging
import os
import os.path as osp
import json
import datetime
import torch
from dataset import LBSNDataset
from sampler import NeighborSampler
from model_next_poi import ModelNextPoi
from model_next_poi import test_step
import random
import numpy as np
from tqdm import tqdm
from conf_parser import Cfg
from torch.utils.tensorboard import SummaryWriter
from model_next_poi import generate_transformer_input


def set_logger(args):
    """
    Write logs to checkpoint and console
    """
    if args.do_train:
        log_file = os.path.join(args.log_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.log_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w+'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_model(model, optimizer, save_variable_list, run_args, argparse_dict):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """
    with open(os.path.join(run_args.log_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(run_args.save_path, 'checkpoint.pt')
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    seed = random.randint(0, 1000000)
    seed_torch(seed)


    cfg = Cfg("run.yml")
    cfg.run_args.seed = seed
    hparam_dict = dict(vars(cfg.run_args).items() | vars(cfg.delta_args).items() | vars(cfg.model_args).items() | vars(
        cfg.dataset_args).items())
    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'tensorboard/{current_time}/{cfg.run_args.dataset}' if cfg.run_args.save_path is None else cfg.run_args.save_path
    log_path = f'log/{current_time}/{cfg.run_args.dataset}' if cfg.run_args.log_path is None else cfg.run_args.log_path
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

    # Initialize dataset
    data_dir = osp.join('data', cfg.run_args.dataset)
    processed_path = osp.join(data_dir, 'processed')
    lbsn_dataset = LBSNDataset(processed_path, cfg.dataset_args)
    logging.info('Data Path: %s' % data_dir)
    logging.info('#num users: %d' % lbsn_dataset.num_users)
    logging.info('#num_pois: %d' % lbsn_dataset.num_pois)
    logging.info('#training_sample: %d' % lbsn_dataset.sample_idx_train.shape[0])
    logging.info('#validation_sample: %d' % lbsn_dataset.sample_idx_valid.shape[0])
    logging.info('#testing_sample: %d' % lbsn_dataset.sample_idx_test.shape[0])
    logging.info(f'Seed: {seed}')

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
        num_workers=cfg.run_args.num_workers,
        shuffle=True,
        pin_memory=True
    )

    cfg.model_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.model_args.num_user = lbsn_dataset.num_users
    cfg.model_args.num_poi = lbsn_dataset.num_pois
    cfg.model_args.num_category = lbsn_dataset.num_category
    cfg.model_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.model_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.model_args.padding_category_id = lbsn_dataset.padding_poi_category
    cfg.model_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.model_args.padding_weekday_id = lbsn_dataset.padding_weekday_id
    cfg.model_args.class_weight = lbsn_dataset.class_weight
    model = ModelNextPoi(cfg.model_args, cfg.delta_args, cfg.run_args, cfg.transformer_args)
    model = model.to(device)
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    logging.info('#Parameters: %s' % count_parameters(model))

    if cfg.run_args.do_train:
        # Set training configuration
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
            logging.info('Loading checkpoint %s...' % cfg.run_args.init_checkpoint)
            checkpoint = torch.load(os.path.join(cfg.run_args.init_checkpoint, 'checkpoint.pt'))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            cooldown_rate = checkpoint['cooldown_rate']
            sizes = checkpoint['sizes']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logging.info('Randomly Initializing Model...')
            init_step = 0
        step = init_step

        # Set valid dataloader as it would be evaluated during training
        logging.info('initial learning rate: %s' % current_learning_rate)

        # Training Loop
        best_metrics = 0.0
        global_step = 0
        for eph in range(cfg.run_args.epoch):
            training_logs = []
            if global_step >= cfg.run_args.max_steps:
                break
            for data in tqdm(sample_result_train):
                model.train()
                split_index = max(data.adjs_t[1].storage.row()).tolist()
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

                if cfg.model_args.do_sequential_transformer:
                    # check-ins input: [20000 * 8]
                    check_in_x = data.x[split_index + 1:]
                    # mask check-ins based on batch: [64 * 20000 * 8]
                    edge_index_tmp = data.adjs_t[0][:, (split_index + 1):].to_dense().detach()
                    checkin_sequential_feature = torch.unsqueeze(
                        edge_index_tmp, dim=-1) * torch.unsqueeze(
                        check_in_x, dim=0).repeat(edge_index_tmp.shape[0], 1, 1)
                    # remove zero tensor and make sequence: [64 * 128 * 8] (embedding layer input)
                    check_in_sequential_input, check_in_sequential_mask = generate_transformer_input(
                        checkin_sequential_feature, model.device, cfg.transformer_args.sequence_length, lbsn_dataset)
                    input_data['sequential_x'] = check_in_sequential_input
                    input_data['sequential_mask'] = check_in_sequential_mask

                out, loss = model(input_data, label=data.y[:, 0])
                training_logs.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                summary_writer.add_scalar(f'train/loss_step', loss, global_step)

                if cfg.run_args.do_valid and global_step % cfg.run_args.valid_steps == 0:
                    logging.info('Evaluating on Valid Dataset...')

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
                        num_workers=cfg.run_args.num_workers,
                        shuffle=False,
                        pin_memory=True
                    )
                    logging.info(f'Epoch {eph} step {global_step} :')
                    recall_res, ndcg_res, map_res, mrr_res, eval_loss = test_step(
                        model,
                        data=sample_result_valid,
                        config=cfg,
                        lbsn_dataset=lbsn_dataset
                    )
                    summary_writer.add_scalar(f'validate/Recall@1', 100*recall_res[1], global_step)
                    summary_writer.add_scalar(f'validate/Recall@5', 100*recall_res[5], global_step)
                    summary_writer.add_scalar(f'validate/Recall@10', 100*recall_res[10], global_step)
                    summary_writer.add_scalar(f'validate/Recall@20', 100*recall_res[20], global_step)
                    summary_writer.add_scalar(f'validate/MRR', mrr_res, global_step)
                    summary_writer.add_scalar(f'validate/eval_loss', eval_loss, global_step)
                    summary_writer.add_scalar('train/learning_rate', current_learning_rate, global_step)

                    metrics = 4 * recall_res[1] + recall_res[20]

                    # save model based on recall@1
                    if metrics > best_metrics:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'cooldown_rate': cfg.run_args.cooldown_rate,
                            'sizes': sizes
                        }
                        logging.info(f'save model at step {global_step} epoch {eph}')
                        save_model(model, optimizer, save_variable_list, cfg.run_args, hparam_dict)
                        best_metrics = metrics

                if global_step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, global_step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * cfg.run_args.cooldown_rate

                if global_step >= cfg.run_args.max_steps:
                    break
                global_step += 1

            epoch_loss = sum([loss for loss in training_logs]) / len(training_logs)
            logging.info('average train loss at step %d is %f:' % (global_step, epoch_loss))
            summary_writer.add_scalar('train/loss_epoch', epoch_loss, eph)

    if cfg.run_args.do_test:
        logging.info('Evaluating on Test Dataset...')
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
            num_workers=cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        checkpoint = torch.load(os.path.join(cfg.run_args.save_path, 'checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        recall_res, ndcg_res, map_res, mrr_res, loss = test_step(model, sample_result_test, cfg, lbsn_dataset)
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
        logging.info(f'Test result : {metric_dict}')
        summary_writer.add_hparams(hparam_dict, metric_dict)
        summary_writer.close()
