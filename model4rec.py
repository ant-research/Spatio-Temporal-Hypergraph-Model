import torch
import torch.nn as nn
from typing import List
from torch import Tensor
import logging
import math
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout, Parameter
from torch.nn.init import xavier_normal_
from torch_sparse import SparseTensor
from conv import DeltaConv
from time_encoder import ContinuousTimeEncoder


class DELTA(nn.Module):
    def __init__(self, N, n_u, n_i, args):
        super(DELTA, self).__init__()
        self.model = args.model.lower()
        self.dropout = args.dropout
        self.N = N
        self.batch_size = args.batch_size
        self.num_neg = args.num_neg
        self.num_users = n_u
        self.num_items = n_i
        self.node_fea_dim = args.embedding_dim
        self.edge_fea_dim = self.node_fea_dim
        self.node_embed_dim = args.embedding_dim
        self.edge_embed_dim = args.embedding_dim
        self.rel_embed_dim = args.embedding_dim
        self.time_embed_dim = args.embedding_dim
        self.num_rel = args.num_rel
        self.num_layers = args.num_layers
        self.attn_heads = args.attn_heads
        self.time_fusion_mode = args.time_fusion_mode
        self.edge_fusion_mode = args.edge_fusion_mode
        self.trans_method = args.trans_method
        self.head_fusion_mode = args.head_fusion_mode
        self.residual_fusion_mode = args.residual_fusion_mode
        self.residual_beta = args.residual_beta
        self.learn_beta = args.learn_beta
        self.with_edge_feature = args.with_edge_feature
        self.is_cuda = args.cuda
        self.activation = args.activation
        if self.activation == 'elu':
            self.act = torch.nn.ELU()
        elif self.activation == 'relu':
            self.act = torch.nn.RReLU()
        elif self.activation == 'leaky_relu':
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.tanh

        def get_param(shape):
            param = Parameter(torch.Tensor(*shape))
            xavier_normal_(param)
            return param

        self.rel_embed = get_param((self.num_rel, self.rel_embed_dim))
        self.x_embed = get_param((self.N, self.node_fea_dim))

        self.edge_attr_lookup = get_param((self.num_rel, self.edge_embed_dim))

        if self.with_edge_feature:
            self.w_edge = get_param((self.edge_fea_dim, self.edge_embed_dim))
            self.b_edge = get_param((self.edge_embed_dim, 1))

        self.cte = ContinuousTimeEncoder(args)
        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()
        if self.model == 'delta':
            self.convs.append(
                DeltaConv((self.node_fea_dim, self.node_embed_dim), self.node_embed_dim, attn_heads=self.attn_heads,
                          residual_beta=self.residual_beta,
                          learn_beta=self.learn_beta,
                          dropout=self.dropout,
                          trans_method=self.trans_method,
                          edge_fusion_mode=self.edge_fusion_mode,
                          time_fusion_mode=self.time_fusion_mode,
                          head_fusion_mode=self.head_fusion_mode,
                          residual_fusion_mode=self.residual_fusion_mode,
                          edge_dim=self.edge_embed_dim,
                          rel_embed_dim=self.rel_embed_dim,
                          time_embed_dim=self.time_embed_dim)
            )
            for i in range(self.num_layers - 1):
                self.convs.append(
                    DeltaConv(self.node_embed_dim, self.node_embed_dim, attn_heads=self.attn_heads,
                              residual_beta=self.residual_beta,
                              learn_beta=self.learn_beta,
                              dropout=self.dropout,
                              trans_method=self.trans_method,
                              edge_fusion_mode=self.edge_fusion_mode,
                              time_fusion_mode=self.time_fusion_mode,
                              head_fusion_mode=self.head_fusion_mode,
                              residual_fusion_mode=self.residual_fusion_mode,
                              edge_dim=self.edge_embed_dim,
                              rel_embed_dim=self.rel_embed_dim,
                              time_embed_dim=self.time_embed_dim)
                )
        for _ in range(self.num_layers):
            self.norms.append(BatchNorm1d(self.node_embed_dim))

    def encode(self, n_id: Tensor, adjs_t: List[SparseTensor], delta_ts: List[Tensor],
               edge_attr: Tensor = None, mode='train') -> Tensor:
        if self.is_cuda:
            n_id = n_id.cuda()
            adjs_t = [adj_t.cuda() for adj_t in adjs_t]
            delta_ts = [delta_t.cuda() for delta_t in delta_ts]
        x = self.x_embed.index_select(dim=0, index=n_id)

        for i, (adj_t, delta_t) in enumerate(zip(adjs_t, delta_ts)):
            edge_index = torch.stack([adj_t.storage.col(), adj_t.storage.row()], dim=0)
            edge_time_embed = self.cte(delta_t)
            edge_type_embed = self.rel_embed.index_select(0, (adj_t.storage.value().type(torch.long).flatten()))
            if self.with_edge_feature:
                edge_attr_embed = edge_attr.matmul(self.w_edge) + self.b_edge.view(1, -1)
            else:
                # for ogb lsc mag data, we generate edge feature through edge_type index lookup
                edge_attr_embed = self.edge_attr_lookup.index_select(0, (
                    adj_t.storage.value().type(torch.long).flatten()))
            x_target = x[:adj_t.size(0)]
            x = self.convs[i]((x, x_target), edge_index=edge_index,
                              edge_attr_embed=edge_attr_embed, edge_time_embed=edge_time_embed,
                              edge_type_embed=edge_type_embed)

            x = self.norms[i](x)
            x = self.act(x)

            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def decode(self, user_batch, item_batch, neg_item_batch, x_user_unique, args, mode='train'):
        if args.cuda:
            user_batch = user_batch.cuda()
            # item_batch = item_batch.cuda()
            neg_item_batch = neg_item_batch.cuda()

        x_user = x_user_unique[user_batch].unsqueeze(1)

        if mode == 'train':
            if args.cuda:
                item_batch = item_batch.cuda()
            negative_sample_size = int(neg_item_batch.size(0) / item_batch.size(0))
            x_neg_item = torch.index_select(
                self.x_embed,
                dim=0,
                index=neg_item_batch.view(-1)
            ).view(-1, negative_sample_size, self.node_embed_dim)

            x_item = torch.index_select(
                self.x_embed,
                dim=0,
                index=item_batch
            ).unsqueeze(1)

            pos_score = (x_user * x_item).sum(dim=2)
            neg_score = (x_user * x_neg_item).sum(dim=2)

        else:
            negative_sample_size = neg_item_batch.size(0)
            x_neg_item = torch.index_select(
                self.x_embed,
                dim=0,
                index=neg_item_batch.view(-1)
            ).view(-1, negative_sample_size, self.node_embed_dim)
            pos_score = None
            neg_score = (x_user * x_neg_item).sum(dim=2)

        return pos_score, neg_score

    @staticmethod
    def train_step(model, optimizer, n_id, adjs_t, delta_ts, user_batch, item_batch, neg_item_batch, args):
        model.train()
        # batch = next(train_iterator)
        optimizer.zero_grad()
        x = model.encode(n_id, adjs_t, delta_ts, mode='train')

        pos_score, neg_score = model.decode(user_batch, item_batch, neg_item_batch, x, args)
        pos_score = F.logsigmoid(pos_score).mean(dim=1)
        neg_score = F.logsigmoid(-neg_score).mean(dim=1)

        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        train_loss = (positive_sample_loss + negative_sample_loss) / 2 + 0.005 * model.x_embed.norm()
        train_loss.backward()
        optimizer.step()
        return train_loss

    @staticmethod
    def test_step(model, n_id, adjs_t, delta_ts, user_batch, item_batch_list, neg_item_batch, args):
        model.eval()
        # batch = batch.to(int(args.device))
        with torch.no_grad():
            if args.cuda:
                user_batch = user_batch.cuda()
                # item_batch_list = item_batch_list.cuda()
                neg_item_batch = neg_item_batch.cuda()
            x = model.encode(n_id, adjs_t, delta_ts, mode='train')

            pos_score, neg_score = model.decode(user_batch, item_batch_list, neg_item_batch, x, args, mode='test')
            argsort = torch.argsort(neg_score, dim=1, descending=True)

            batch_size_eval = x.size(0)

            for i in range(batch_size_eval):
                ranking = argsort[i, :]
                item_top = ranking[:50]
                item_top += 1
                item_top = item_top.cpu().numpy().tolist()

                iid_list = item_batch_list[i].tolist()  # gt item
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(item_top):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                recall_tmp = recall * 1.0 / len(true_item_set)
                ndcg_tmp = dcg / idcg if recall > 0 else 0.0
                hitrate_tmp = 1 if recall > 0 else 0

        return recall_tmp, ndcg_tmp, hitrate_tmp
