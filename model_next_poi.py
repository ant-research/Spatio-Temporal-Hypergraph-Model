import numpy as np
import logging
import torch
import tqdm

from conv import DeltaConv
from time_encoder import ContinuousTimeEncoder
from time_encoder import ContinuousDistEncoder
from time_encoder import ContinuousDistEncoderHSTLSTM
from time_encoder import ContinuousDistEncoderSimple
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
import math


class ModelNextPoi(torch.nn.Module):
    def __init__(self, args, delta_args, run_args, transformer_args):
        super().__init__()
        self.device = run_args.device
        self.batch_size = run_args.batch_size
        self.eval_batch_size = run_args.eval_batch_size
        self.learning_rate = run_args.learning_rate
        self.do_traj2traj = args.do_traj2traj
        self.do_target_attention = args.do_target_attention
        self.distance_encoder_type = args.distance_encoder_type
        self.do_positional_encoding = transformer_args.do_positional_encoding
        self.do_sequential_transformer = args.do_sequential_transformer
        self.dropout_rate = args.dropout_rate
        self.num_conv_layers = args.num_conv_layers
        self.num_user = args.num_user
        self.num_poi = args.num_poi
        self.num_category = args.num_category
        self.num_dayofweek = 7
        self.num_hourofday = 24
        self.num_edge_type = args.num_edge_type
        self.generate_edge_attr = args.generate_edge_attr
        self.embedding_dim = args.embedding_dim
        self.st_embedding_dim = args.st_embedding_dim
        self.embedding_fusion_type = args.embedding_fusion_type
        self.checkin_embedding_layer = ModelCheckinEmbedding(
            self.embedding_dim,
            self.embedding_fusion_type,
            self.num_user,
            self.num_poi,
            self.num_category,
            self.num_dayofweek,
            self.num_hourofday,
            args.padding_user_id,
            args.padding_poi_id,
            args.padding_category_id,
            args.padding_weekday_id,
            args.padding_hour_id
        )
        self.edge_type_embedding_layer = ModelEdgeEmbedding(
            self.checkin_embedding_layer.output_embedding_size, self.embedding_fusion_type, self.num_edge_type
        )

        if args.activation == 'elu':
            self.act = torch.nn.ELU()
        elif args.activation == 'relu':
            self.act = torch.nn.RReLU()
        elif args.activation == 'leaky_relu':
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.tanh

        if delta_args.time_fusion_mode == 'add':
            continuous_encoder_dim = self.checkin_embedding_layer.output_embedding_size
        else:
            continuous_encoder_dim = self.st_embedding_dim

        if self.generate_edge_attr:
            # use edge_type to create edge_attr_embed
            self.edge_attr_embedding_layer = ModelEdgeEmbedding(
                self.checkin_embedding_layer.output_embedding_size, self.embedding_fusion_type, self.num_edge_type
            )
        else:
            # source_traj_size, target_traj_size, jaccard similarity as the raw features, and do MLP
            if delta_args.edge_fusion_mode == 'add':
                self.edge_attr_embedding_layer = torch.nn.Linear(3, self.checkin_embedding_layer.output_embedding_size)
            else:
                self.edge_attr_embedding_layer = None

        self.conv_list = torch.nn.ModuleList()
        self.conv_for_time_filter = DeltaConv(
            in_channels=self.checkin_embedding_layer.output_embedding_size,
            out_channels=self.checkin_embedding_layer.output_embedding_size,
            attn_heads=delta_args.num_attention_heads,
            residual_beta=delta_args.residual_beta,
            learn_beta=delta_args.learn_beta,
            dropout=delta_args.conv_dropout_rate,
            trans_method=delta_args.trans_method,
            edge_fusion_mode=delta_args.edge_fusion_mode,
            time_fusion_mode=delta_args.time_fusion_mode,
            head_fusion_mode=delta_args.head_fusion_mode,
            residual_fusion_mode=None,
            edge_dim=None,
            rel_embed_dim=self.checkin_embedding_layer.output_embedding_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=delta_args.negative_slope,
            have_query_feature=False
        )
        self.norms_for_time_filter = torch.nn.BatchNorm1d(self.checkin_embedding_layer.output_embedding_size)
        self.dropout_for_time_filter = torch.nn.Dropout(self.dropout_rate)
        for i in range(self.num_conv_layers):
            if i == 0:
                # ci2traj full
                have_query_feature = False
                residual_fusion_mode = None
                edge_dim = None
            else:
                # traj2traj
                have_query_feature = True
                residual_fusion_mode = delta_args.residual_fusion_mode
                if self.edge_attr_embedding_layer is None:
                    edge_dim = 3
                else:
                    edge_dim = self.checkin_embedding_layer.output_embedding_size
            self.conv_list.append(
                DeltaConv(
                    in_channels=self.checkin_embedding_layer.output_embedding_size,
                    out_channels=self.checkin_embedding_layer.output_embedding_size,
                    attn_heads=delta_args.num_attention_heads,
                    residual_beta=delta_args.residual_beta,
                    learn_beta=delta_args.learn_beta,
                    dropout=delta_args.conv_dropout_rate,
                    trans_method=delta_args.trans_method,
                    edge_fusion_mode=delta_args.edge_fusion_mode,
                    time_fusion_mode=delta_args.time_fusion_mode,
                    head_fusion_mode=delta_args.head_fusion_mode,
                    residual_fusion_mode=residual_fusion_mode,
                    edge_dim=edge_dim,
                    rel_embed_dim=self.checkin_embedding_layer.output_embedding_size,
                    time_embed_dim=continuous_encoder_dim,
                    dist_embed_dim=continuous_encoder_dim,
                    negative_slope=delta_args.negative_slope,
                    have_query_feature=have_query_feature
                )
            )
        self.norms_list = torch.nn.ModuleList()
        for i in range(self.num_conv_layers):
            self.norms_list.append(torch.nn.BatchNorm1d(self.checkin_embedding_layer.output_embedding_size))

        self.dropout_list = torch.nn.ModuleList()
        for i in range(self.num_conv_layers):
            self.dropout_list.append(torch.nn.Dropout(self.dropout_rate))

        # 必须包含embedding_dim,phase_factor,use_linear_trans
        self.continuous_time_encoder = ContinuousTimeEncoder(
            args, continuous_encoder_dim)

        if args.distance_encoder_type == 'stan':
            self.continuous_distance_encoder = ContinuousDistEncoder(args, continuous_encoder_dim, args.spatial_slots)
        elif args.distance_encoder_type == 'time':
            self.continuous_distance_encoder = ContinuousTimeEncoder(
                args, continuous_encoder_dim)
        elif args.distance_encoder_type == 'hstlstm':
            self.continuous_distance_encoder = ContinuousDistEncoderHSTLSTM(args, continuous_encoder_dim, args.spatial_slots)
        elif args.distance_encoder_type == 'simple':
            self.continuous_distance_encoder = ContinuousDistEncoderSimple(args, continuous_encoder_dim, args.spatial_slots)
        else:
            raise ValueError("Get wrong distance_encoder_type argument!")

        if self.do_sequential_transformer:
            encoder_layers = TransformerEncoderLayer(
                d_model=self.checkin_embedding_layer.output_embedding_size,
                nhead=transformer_args.header_num,
                dim_feedforward=transformer_args.hidden_size,
                dropout=transformer_args.dropout
            )
            self.transformer_encoder = TransformerEncoder(
                encoder_layers,
                num_layers=transformer_args.encoder_layers_num
            )
            self.transformer_positional_encoding = PositionalEncoding(
                self.checkin_embedding_layer.output_embedding_size, self.device, transformer_args.dropout)

            if args.do_target_attention:
                self.target_attention = torch.nn.MultiheadAttention(
                    self.checkin_embedding_layer.output_embedding_size, 1)
                self.fusion_head = torch.nn.Linear(self.checkin_embedding_layer.output_embedding_size, self.num_poi)
            else:
                self.fusion_head = torch.nn.Linear(
                    self.checkin_embedding_layer.output_embedding_size * 2, self.num_poi)
        else:
            self.fusion_head = torch.nn.Linear(self.checkin_embedding_layer.output_embedding_size, self.num_poi)

        if args.do_weighted_loss:
            self.loss_func = torch.nn.CrossEntropyLoss(weight=args.class_weight)
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, data, label=None, mode='train'):
        # 输入包含x, edge_index, edge_type, delta_ts, delta_ss
        input_x = data['x']  # 24 * 8
        split_idx = data['split_index']
        # split_idx = max(data['edge_index'][1].storage.row()).tolist()  # 找到checkin和trajectory切分点

        check_in_x = input_x[split_idx+1:]  # checkin, 16 * 8
        # embedding_input = [check_in_x[:, i] for i in range(3)]  # [16 * 8]
        checkin_feature = self.checkin_embedding_layer(check_in_x)
        trajectory_feature = torch.zeros(
            split_idx+1, self.checkin_embedding_layer.output_embedding_size, device=checkin_feature.device)
        x = torch.cat([trajectory_feature, checkin_feature], dim=0)  # 24 * 128

        edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
        if self.distance_encoder_type == 'stan':
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='ch2tj')
        else:
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

        edge_attr_embed, edge_type_embed = None, None
        if data['edge_type'][0] is not None:
            if self.generate_edge_attr:
                edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
            edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

        x_for_time_filter = self.conv_for_time_filter(
            x,
            edge_index=data['edge_index'][0],
            edge_attr_embed=edge_attr_embed,
            edge_time_embed=edge_time_embed,
            edge_dist_embed=edge_distance_embed,
            edge_type_embed=edge_type_embed
        )
        x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
        x_for_time_filter = self.act(x_for_time_filter)
        x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

        if data['edge_index'][-1] is not None and self.do_traj2traj:
            # all conv
            for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
                    zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:],
                        data["edge_type"][1:])
            ):
                edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
                if self.distance_encoder_type == 'stan':
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='tj2tj')
                else:
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis)

                edge_attr_embed, edge_type_embed = None, None
                if edge_type is not None:
                    edge_type_embed = self.edge_type_embedding_layer(edge_type)
                    if self.generate_edge_attr:
                        edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
                    else:
                        if self.edge_attr_embedding_layer:
                            edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
                        else:
                            edge_attr_embed = edge_attr.to(torch.float32)

                if idx == len(data['edge_index']) - 2:
                    if mode in ('test', 'validate'):
                        batch_size = self.eval_batch_size
                    else:
                        batch_size = self.batch_size
                    x_target = x_for_time_filter[:batch_size]
                else:
                    x_target = x[:edge_index.sparse_sizes()[0]]
                x = self.conv_list[idx](
                    (x, x_target),
                    edge_index=edge_index,
                    edge_attr_embed=edge_attr_embed,
                    edge_time_embed=edge_time_embed,
                    edge_dist_embed=edge_distance_embed,
                    edge_type_embed=edge_type_embed
                )
                x = self.norms_list[idx](x)
                x = self.act(x)
                x = self.dropout_list[idx](x)
        else:
            x = x_for_time_filter

        if self.do_sequential_transformer:
            sequential_feature = self.checkin_embedding_layer(data['sequential_x'])
            if self.do_positional_encoding:
                self.transformer_positional_encoding(sequential_feature)
            sequential_feature = sequential_feature.transpose(1, 0)
            sequential_out = self.transformer_encoder(sequential_feature, src_key_padding_mask=data['sequential_mask'])
            if not self.do_target_attention:
                sequential_out = torch.mean(sequential_out, dim=0)
                x = self.fusion_head(torch.cat([sequential_out, x], dim=-1))
            else:
                x = self.target_attention(
                    query=torch.unsqueeze(x, 0), key=sequential_out, value=sequential_out,
                    key_padding_mask=data['sequential_mask']
                )
                x = self.fusion_head(torch.squeeze(x[0]))

        x = self.fusion_head(x)

        if label is not None:
            loss = self.loss_func(x, label.long())
        else:
            loss = None
        return x, loss


class ModelCheckinEmbedding(torch.nn.Module):
    def __init__(
            self,
            embedding_size,
            fusion_type,
            num_user, num_poi, num_category, num_dayofweek, num_hourofday,
            user_padding_id, poi_padding_id, category_padding_id, weekday_padding_id, hour_padding_id
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.fusion_type = fusion_type
        self.user_embedding = torch.nn.Embedding(
            num_user + 1, self.embedding_size, padding_idx=user_padding_id)
        self.poi_embedding = torch.nn.Embedding(
            num_poi + 1, self.embedding_size, padding_idx=poi_padding_id)
        self.category_embedding = torch.nn.Embedding(
            num_category + 1, self.embedding_size, padding_idx=category_padding_id)
        self.dayofweek_embedding = torch.nn.Embedding(
            num_dayofweek + 1, self.embedding_size, padding_idx=weekday_padding_id)
        self.hourofday_embedding = torch.nn.Embedding(
            num_hourofday + 1, self.embedding_size, padding_idx=hour_padding_id)
        if self.fusion_type == 'concat':
            self.output_embedding_size = 5 * self.embedding_size
        elif self.fusion_type == 'add':
            self.output_embedding_size = embedding_size
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")

    def forward(self, data):
        embedding_list = [
            self.user_embedding(data[..., 0].long()),
            self.poi_embedding(data[..., 1].long()),
            self.category_embedding(data[..., 2].long()),
            self.dayofweek_embedding(data[..., 6].long()),
            self.hourofday_embedding(data[..., 7].long())
        ]
        if self.fusion_type == 'concat':
            self.output_embedding_size = len(embedding_list) * self.embedding_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return torch.squeeze(sum(embedding_list))
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")


class ModelEdgeEmbedding(torch.nn.Module):
    def __init__(self, embedding_size, fusion_type, num_edge_type):
        super().__init__()
        self.embedding_size = embedding_size
        self.fusion_type = fusion_type
        self.edge_type_embedding = torch.nn.Embedding(num_edge_type, self.embedding_size)
        self.output_embedding_size = self.embedding_size

    def forward(self, data):
        embedding_list = [self.edge_type_embedding(data.long())]

        if self.fusion_type == 'concat':
            self.output_embedding_size = len(embedding_list) * self.embedding_size
            return torch.cat(embedding_list, -1)
        elif self.fusion_type == 'add':
            return sum(embedding_list)
        else:
            raise ValueError(f"Get wrong fusion type {self.fusion_type}")


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def generate_transformer_input(
        sequential_feature,
        device,
        max_length,
        dataset
):
    """
    generate transformer input
    """
    padding_tensor = torch.tensor([
        dataset.padding_poi_id,
        dataset.padding_poi_category,
        dataset.padding_user_id,
        0,
        0,
        0,
        dataset.padding_hour_id,
        dataset.padding_weekday_id
    ], dtype=torch.float64, device=device)
    input_ids = torch.unsqueeze(torch.unsqueeze(padding_tensor, dim=0).repeat(max_length, 1), dim=0).repeat(
        sequential_feature.size()[0], 1, 1)
    mask = torch.ones(torch.Size([sequential_feature.size()[0], max_length]), dtype=torch.bool, device=device)
    nonzero_index = torch.nonzero(sequential_feature)
    nonzero_index_tmp = nonzero_index[:, :2].unique(dim=0).cpu().detach().numpy()
    sequential_time_feature = [
        (m, n, sequential_feature[m, n, 3].cpu().detach().numpy().tolist()) for m, n in nonzero_index_tmp]
    # TODO: truncate check-in backwards
    nonzero_index_tmp = sorted(sequential_time_feature, key=lambda x: (x[0], -x[-1]))
    batch_len = 0
    batch_idx_tmp = 0
    nonzero_index_tmp_copy = []
    for m, n, t in nonzero_index_tmp:
        if batch_idx_tmp != m:
            batch_idx_tmp += 1
            batch_len = 0
        if batch_len >= max_length:
            continue
        nonzero_index_tmp_copy.append((m, n, t))
        batch_len += 1

    batch_len = 0
    batch_idx_tmp = 0
    nonzero_index_tmp_copy = sorted(nonzero_index_tmp_copy, key=lambda x: (x[0], x[-1]))
    for m, n, _ in nonzero_index_tmp_copy:
        if batch_idx_tmp != m:
            batch_idx_tmp += 1
            batch_len = 0
        if batch_len >= max_length:
            continue
        input_ids[batch_idx_tmp, batch_len] = sequential_feature[m, n]
        mask[batch_idx_tmp, batch_len] = False
        batch_len += 1
    return input_ids, mask


def test_step(model, data, config, lbsn_dataset, ks=(1, 5, 10, 20)):

    def calc_recall(lab, prd, k):
        return torch.sum(torch.sum(lab == prd[:, :k], dim=1)) / lab.shape[0]

    def calc_ndcg(lab, prd, k):
        exist_pos = (prd[:, :k] == lab).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos.float() + 1)
        return torch.sum(ndcg) / lab.shape[0]

    def calc_map(lab, prd, k):
        exist_pos = (prd[:, :k] == lab).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / lab.shape[0]

    def calc_mrr(lab, prd):
        exist_pos = (prd == lab).nonzero()[:, 1] + 1
        mrr = 1 / exist_pos
        return torch.sum(mrr) / lab.shape[0]

    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for row in tqdm.tqdm(data):
            split_index = max(row.adjs_t[1].storage.row()).tolist()
            row = row.to(model.device)

            input_data = {
                'x': row.x,
                'edge_index': row.adjs_t,
                'edge_attr': row.edge_attrs,
                'split_index': split_index,
                'delta_ts': row.edge_delta_ts,
                'delta_ss': row.edge_delta_ss,
                'edge_type': row.edge_types
            }

            if config.model_args.do_sequential_transformer:
                # check-ins input: [20000 * 8]
                check_in_x = row.x[split_index + 1:]
                # mask check-ins based on batch: [64 * 20000 * 8]
                edge_index_tmp = row.adjs_t[0][:, (split_index + 1):].to_dense()
                checkin_sequential_feature = torch.unsqueeze(
                    edge_index_tmp, dim=-1) * torch.unsqueeze(
                    check_in_x, dim=0).repeat(edge_index_tmp.shape[0], 1, 1)
                # remove zero tensor and make sequence: [64 * 128 * 8] (embedding layer input)
                check_in_sequential_input, check_in_sequential_mask = generate_transformer_input(
                    checkin_sequential_feature, model.device, config.transformer_args.sequence_length, lbsn_dataset)
                input_data['sequential_x'] = check_in_sequential_input
                input_data['sequential_mask'] = check_in_sequential_mask

            out, loss = model(input_data, label=row.y[:, 0], mode='test')
            loss_list.append(loss.cpu().detach().numpy().tolist())
            ranking = torch.sort(out, descending=True)[1]
            pred_list.append(ranking.cpu().detach())
            label_list.append(row.y[:, :1].cpu())
    pred_ = torch.cat(pred_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    recalls, NDCGs, MAPs = {}, {}, {}
    logging.info(f"Average loss: {np.mean(loss_list)}")
    for k_ in ks:
        recalls[k_] = calc_recall(label_, pred_, k_).cpu().detach().numpy().tolist()
        NDCGs[k_] = calc_ndcg(label_, pred_, k_).cpu().detach().numpy().tolist()
        MAPs[k_] = calc_map(label_, pred_, k_).cpu().detach().numpy().tolist()
        logging.info(f"Recall@{k_} : {recalls[k_]},\tNDCG@{k_} : {NDCGs[k_]},\tMAP@{k_} : {MAPs[k_]}")
    mrr_res = calc_mrr(label_, pred_).cpu().detach().numpy().tolist()
    logging.info(f"MRR : {mrr_res}")
    return recalls, NDCGs, MAPs, mrr_res, np.mean(loss_list)
