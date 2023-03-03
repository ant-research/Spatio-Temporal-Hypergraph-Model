import torch
from torch import nn
import numpy as np
from utils import cal_slot_distance_batch
import argparse
from dataset import LBSNDataset


class ContinuousTimeEncoder(nn.Module):
    r"""
    This is a trainable encoder to map continuous time value into a low-dimension time vector.
    Ref: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/module.py

    The input of ts should be like [E, 1] with all time interval as values.
    """

    def __init__(self, args, embedding_dim):
        super(ContinuousTimeEncoder, self).__init__()
        self.time_dim = embedding_dim
        self.expand_dim = self.time_dim
        self.factor = args.phase_factor
        self.use_linear_trans = args.use_linear_trans
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())
        if self.use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [E, 1]
        if ts.dim() == 1:
            dim = 1
            edge_len = ts.size().numel()
        else:
            edge_len, dim = ts.size()
        ts = ts.view(edge_len, dim)  # [E, 1]
        map_ts = ts * self.basis_freq.view(1, -1)  # [E, time_dim]
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        if self.use_linear_trans:
            harmonic = harmonic.type(self.dense.weight.dtype)
            harmonic = self.dense(harmonic)
        return harmonic


class ContinuousDistEncoderHSTLSTM(nn.Module):
    r"""
    This is a trainable encoder to map continuous distance value into a low-dimension vector.

    The input of ts should be like [E, 1] with all dist interval as values.
    """

    def __init__(self, args, embedding_dim, spatial_slots):
        super(ContinuousDistEncoderHSTLSTM, self).__init__()
        self.dist_dim = embedding_dim
        self.spatial_slots = spatial_slots
        self.embed_q = nn.Embedding(len(spatial_slots), self.dist_dim)
        self.device = args.gpu

    def place_parameters(self, ld, hd, l, h):
        if self.device == 'cpu':
            ld = torch.from_numpy(np.array(ld)).type(torch.FloatTensor)
            hd = torch.from_numpy(np.array(hd)).type(torch.FloatTensor)
            l = torch.from_numpy(np.array(l)).type(torch.LongTensor)
            h = torch.from_numpy(np.array(h)).type(torch.LongTensor)
        else:     
            ld = torch.from_numpy(np.array(ld, dtype=np.float16)).type(torch.FloatTensor).to(self.device)
            hd = torch.from_numpy(np.array(hd, dtype=np.float16)).type(torch.FloatTensor).to(self.device)
            l = torch.from_numpy(np.array(l, dtype=np.float16)).type(torch.LongTensor).to(self.device)
            h = torch.from_numpy(np.array(h, dtype=np.float16)).type(torch.LongTensor).to(self.device)
        return ld, hd, l, h

    def cal_inter(self, ld, hd, l, h, embed):
        """
        Calculate a linear interpolation.
        :param ld: Distances to lower bound, shape (batch_size, step)
        :param hd: Distances to higher bound, shape (batch_size, step)
        :param l: Lower bound indexes, shape (batch_size, step)
        :param h: Higher bound indexes, shape (batch_size, step)
        """
        # Fetch the embed of higher and lower bound.
        # Each result shape (batch_size, step, input_size)
        l_embed = embed(l)
        h_embed = embed(h)
        return torch.stack([hd], -1) * l_embed + torch.stack([ld], -1) * h_embed

    def forward(self, dist):
        self.spatial_slots = sorted(self.spatial_slots)  # when num_slots=5000, interval=2.56; num_slots=2000, interval=6.4
        d_ld, d_hd, d_l, d_h = self.place_parameters(*cal_slot_distance_batch(dist, self.spatial_slots))
        # d_ld, d_hd, d_l, d_h = cal_slot_distance_batch(dist, self.spatial_slots)
        batch_q = self.cal_inter(d_ld, d_hd, d_l, d_h, self.embed_q)
        return batch_q


class ContinuousDistEncoder(nn.Module):
    r"""
    This is a trainable encoder to map continuous distance value into a low-dimension vector.

    The input of ts should be like [E, 1] with all dist interval as values.
    """

    def __init__(self, args, embedding_dim, spatial_slots):
        super(ContinuousDistEncoder, self).__init__()
        self.dist_dim = embedding_dim
        self.min_d, self.max_d_ch2tj, self.max_d_tj2tj = spatial_slots
        self.embed_min = nn.Embedding(1, self.dist_dim)
        self.embed_max = nn.Embedding(1, self.dist_dim)
        self.embed_max_traj = nn.Embedding(1, self.dist_dim)
        self.quantile = args.quantile

    def forward(self, dist, dist_type):
        if dist_type == 'ch2tj':
            emb_low, emb_high = self.embed_min.weight, self.embed_max.weight
            max_d = self.max_d_ch2tj    # tensor(43.7633)
        else:
            emb_low, emb_high = self.embed_min.weight, self.embed_max_traj.weight
            max_d = self.max_d_tj2tj

        # dist = dist.clip(0, torch.quantile(max_d, self.quantile))
        # dist_max_clip = torch.quantile(dist, self.quantile)
        dist = dist.clip(0, max_d)
        vsl, vsu = (dist - self.min_d).unsqueeze(-1).expand(-1, self.dist_dim), \
                             (max_d - dist).unsqueeze(-1).expand(-1, self.dist_dim)

        space_interval = (emb_low * vsu + emb_high * vsl) / (max_d - self.min_d)
        return space_interval

class ContinuousDistEncoderSimple(nn.Module):
    r"""
    This is a trainable encoder to map continuous distance value into a low-dimension vector.

    The input of ts should be like [E, 1] with all dist interval as values.
    """

    def __init__(self, args, embedding_dim, spatial_slots):
        super(ContinuousDistEncoderSimple, self).__init__()
        self.dist_dim = embedding_dim
        self.min_d, self.max_d, self.max_d_traj = spatial_slots
        self.embed_unit = nn.Embedding(1, self.dist_dim)

    def forward(self, dist):
        dist = dist.unsqueeze(-1).expand(-1, self.dist_dim)
        return dist * self.embed_unit.weight


# Module Test
class DELTA(nn.Module):
    def __init__(self,spatial_slots, args):
        super(DELTA, self).__init__()
        self.cde = ContinuousDistEncoder(args, spatial_slots)

    def forward(self, batch):
        edge_dist_embed = self.cde(batch)
        return edge_dist_embed

    @staticmethod
    def train_step(model, batch, optimizer):
        model.train()
        edge_dist_embed = model(batch)
        print(edge_dist_embed.shape)
        train_loss = edge_dist_embed.norm()
        train_loss.backward()
        optimizer.step()
        return train_loss

    @staticmethod
    def test_step(model):
        model.eval()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--data_dir', type=str, default='./data/tky/processed/tky')
    parser.add_argument('--mode', type=str, default='session')
    parser.add_argument('--sizes', type=str, default='10-10')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--num_neg', type=int, default=2)
    parser.add_argument('--feature_dim', type=int, default=100)
    parser.add_argument('--num_spatial_slots', type=int, default=100)
    parser.add_argument('--spatial_slot_type', type=str, default='linear')
    parser.add_argument('--embedding_dim', type=int, default=128)

    args = parser.parse_args()
    mydataset = LBSNDataset(args.data_dir, args)
    model = DELTA(mydataset.spatial_slots, args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    sample = torch.tensor([130000.,78000,140000,110000,79000])
    loss = model.train_step(model, sample, optimizer)
    print(loss)
