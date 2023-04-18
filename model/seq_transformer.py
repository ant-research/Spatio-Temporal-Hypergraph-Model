import torch
from torch import nn
from layer import CheckinEmbedding, PositionEncoder


class SequentialTransformer(nn.Module):
    def __init__(self, cfg):
        super(SequentialTransformer, self).__init__()
        self.dataset_args = cfg.dataset_args
        self.device = cfg.run_args.device
        self.do_positional_encoding = cfg.seq_transformer_args.do_positional_encoding
        self.embed_fusion_type = cfg.model_args.embed_fusion_type
        self.checkin_embedding_layer = CheckinEmbedding(
            embed_size=cfg.model_args.embed_size,
            fusion_type=self.embed_fusion_type,
            dataset_args=self.dataset_args
        )
        self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.checkin_embed_size,
            nhead=cfg.seq_transformer_args.header_num,
            dim_feedforward=cfg.seq_transformer_args.hidden_size,
            dropout=cfg.seq_transformer_args.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=cfg.seq_transformer_args.encoder_layers_num
        )
        self.transformer_positional_encoding = PositionEncoder(
            self.checkin_embed_size,
            self.device,
            cfg.seq_transformer_args.dropout
        )
        self.sequence_length = cfg.seq_transformer_args.sequence_length

        self.linear = torch.nn.Linear(self.checkin_embed_size, self.dataset_args.num_poi)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, data, label=None, mode='train'):
        # Generate seq input
        split_idx = data['split_index']
        check_in_x = data['x'][split_idx + 1:]

        # mask checkins based on batch: [N, #checkin, d]
        edge_index_tmp = data['edge_index'][0][:, (split_idx + 1):].to_dense()
        checkin_sequential_feature = torch.unsqueeze(edge_index_tmp, dim=-1) * torch.unsqueeze(
            check_in_x, dim=0).repeat(edge_index_tmp.shape[0], 1, 1)

        # remove zero tensor and make sequence: [N, #checkin, d] (embedding layer input)
        check_in_sequential_input, check_in_sequential_mask = self.generate_sequential_input(
            checkin_sequential_feature,
            device=check_in_x.device,
            max_length=self.sequence_length
        )

        sequential_feature = self.checkin_embedding_layer(check_in_sequential_input)
        if self.do_positional_encoding:
            self.transformer_positional_encoding(sequential_feature)
        sequential_feature = sequential_feature.transpose(1, 0)
        sequential_out = self.transformer_encoder(sequential_feature, src_key_padding_mask=check_in_sequential_mask)
        sequential_out = torch.mean(sequential_out, dim=0)

        logits = self.linear(sequential_out)
        loss = self.loss_func(logits, label.long())
        return logits, loss

    def generate_sequential_input(self, sequential_feature, device, max_length):
        """
        Generate sequential input for sequential model
        """
        padding_tensor = torch.tensor([
            self.dataset_args.padding_user_id,
            self.dataset_args.padding_poi_id,
            self.dataset_args.padding_poi_category,
            0,
            0,
            0,
            self.dataset_args.padding_weekday_id,
            self.dataset_args.padding_hour_id
        ], dtype=torch.float64, device=device)
        input_ids = torch.unsqueeze(torch.unsqueeze(padding_tensor, dim=0).repeat(max_length, 1), dim=0).repeat(
            sequential_feature.size()[0], 1, 1)
        mask = torch.ones(torch.Size([sequential_feature.size()[0], max_length]), dtype=torch.bool, device=device)
        nonzero_index = torch.nonzero(sequential_feature)
        nonzero_index_tmp = nonzero_index[:, :2].unique(dim=0).cpu().detach().numpy()
        sequential_time_feature = [
            (m, n, sequential_feature[m, n, 3].cpu().detach().numpy().tolist()) for m, n in nonzero_index_tmp]

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
