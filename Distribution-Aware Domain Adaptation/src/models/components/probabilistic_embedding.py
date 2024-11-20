from functools import partial

import torch
from torch import nn
from torch.nn import LayerNorm, MultiheadAttention, Sigmoid
from torch.nn import functional as F


class MeanComputation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mha = MultiheadAttention(dim, 1)
        self.sigmoid = Sigmoid()
        self.norm = LayerNorm(dim)
        self.l2 = partial(F.normalize, p=2, dim=-1)

    def forward(self, features):
        local_attn, _ = self.mha(features, features, features)
        local_attn = self.sigmoid(local_attn)

        output = features + local_attn
        output = self.norm(output)
        output = self.l2(output)
        return output


class VarianceComputation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mha = MultiheadAttention(dim, 1)
        self.sigmoid = Sigmoid()
        self.norm = LayerNorm(dim)
        self.l2 = partial(F.normalize, p=2, dim=-1)

    def forward(self, features):
        local_attn, _ = self.mha(features, features, features)
        output = features + local_attn
        return output


class ProbabilisticEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, n_samples, firstk=-1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_samples = n_samples
        self.firstk = firstk
        # self.mu_layer = MeanComputation(in_dim)
        # self.logsigma_layer = VarianceComputation(in_dim)
        self.mu_layer = nn.Linear(in_dim, out_dim)
        self.logsigma_layer = nn.Linear(in_dim, out_dim)

    @staticmethod
    def _sample_gaussian_tensors(mu, logsigma, n_samples, sigma_scale=1.0):
        # mu/logsigma (bsize, maxlen, dim)
        # -> samples (bsize, n_samples, maxlen, dim)
        if n_samples == 0:
            return mu.unsqueeze(1)
        else:
            bsize, *other_sizes = tuple(mu.size())
            assert bsize == logsigma.size(0) and tuple(other_sizes) == tuple(logsigma.shape[1:])
            eps = torch.randn(bsize, n_samples, *other_sizes, dtype=mu.dtype, device=mu.device)
            sigma = torch.exp(logsigma.unsqueeze(1))
            sigma = sigma * sigma_scale
            samples = eps.mul(sigma).add_(mu.unsqueeze(1))
            return samples

    def forward(self, features):
        # 1. truncate
        if self.firstk > 0:
            features = features[:, : self.firstk, :]
        # 2. modeling probabilistic embedding (gaussian distribution)
        mu = self.mu_layer(features)
        logsigma = self.logsigma_layer(features)
        # mu, logsigma: (``bsize, firstk, dim``)
        # sampling features
        samples = self._sample_gaussian_tensors(
            mu=mu, logsigma=logsigma, n_samples=self.n_samples, sigma_scale=1.0
        )
        return samples
