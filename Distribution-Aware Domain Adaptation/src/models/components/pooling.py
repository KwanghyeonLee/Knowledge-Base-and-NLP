import copy
from typing import Any, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, einsum, nn
from torch.nn import functional as F

from src.models.components.unet_parts import DoubleConv, Down, OutConv, Up


class BasePooling(nn.Module):
    def __init__(self, pooling_target: str = "cls"):
        super().__init__()
        self.pooling_target = pooling_target
        assert pooling_target in ["cls", "mean"]

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Any]:
        output = None
        if self.pooling_target == "cls":
            output = token_embeddings[:, 0, :]
        elif self.pooling_target == "mean":
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output = sum_embeddings / sum_mask

        return output, None


class AttentionPooling(nn.Module):
    def __init__(self, vocab_size: int = 768):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(vocab_size, vocab_size),
            nn.Tanh(),
            nn.Linear(vocab_size, 1),
        )
        for m in self.attn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Any]:
        attn_logits = self.attn(token_embeddings).squeeze(2)  # (b, smth_len, 1) -> (b, smth_len)
        attn_logits[attention_mask == 0] = float("-inf")
        attn_weights = attn_logits.softmax(dim=-1)

        output = einsum("bsv,bs->bv", token_embeddings, attn_weights)

        return output, attn_logits


class AttentionDropoutPooling(nn.Module):
    def __init__(self, vocab_size: int = 768, dropout: float = 0.25):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(vocab_size, vocab_size),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(vocab_size, 1),
        )
        for m in self.attn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Any]:
        attn_logits = self.attn(token_embeddings).squeeze(2)  # (b, smth_len, 1) -> (b, smth_len)
        attn_logits[attention_mask == 0] = float("-inf")
        attn_weights = attn_logits.softmax(dim=-1)

        output = einsum("bsv,bs->bv", token_embeddings, attn_weights)

        return output, attn_logits


class AdaptivePooling(nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.balance = nn.Linear(d_hidden, 1)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.0) / np.sqrt(self.linear.in_features + self.linear.out_features)
        self.linear.weight.data.uniform_(-r, r)
        self.linear.bias.data.fill_(0)

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Any]:
        mask = repeat(attention_mask, "batch len -> batch len tmp", tmp=1)
        mask_features = token_embeddings.masked_fill(mask == 0, -1000)
        sorted_mask_features = mask_features.sort(dim=1, descending=True).values

        # embeddings = token_embeddings
        # embeddings[attention_mask == 0] = float(0)

        # embedding-level
        embed_weights = F.softmax(sorted_mask_features, dim=1)
        embed_features = (mask_features * embed_weights).sum(1)

        # token-level
        # token_weights = [B x K x D]
        mask_features = mask_features.masked_fill(mask == 0, 0)
        token_weights = self.linear(mask_features)
        token_weights = F.softmax(self.relu(token_weights), dim=1)
        token_features = (mask_features * token_weights).sum(dim=1)

        # fusion
        fusion_features = torch.cat(
            [token_features.unsqueeze(1), embed_features.unsqueeze(1)], dim=1
        )
        fusion_weights = F.softmax(self.balance(fusion_features), dim=1)
        pool_features = (fusion_features * fusion_weights).sum(1)

        # fusion_weights.squeeze()
        return pool_features, None


class LSTMPooling(nn.Module):
    def __init__(self, vocab_size: int = 768):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=vocab_size,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )

    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Any]:
        result, (hn, cn) = self.lstm(token_embeddings)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(result.size()).float()
        sum_embeddings = torch.sum(result * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output = sum_embeddings / sum_mask
        # logits = rearrange(hn, "layer batch len -> (layer batch) len")
        return output, None


class UNetPooling(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.dropout = nn.Dropout(p=0.1)

        self.inc = DoubleConv(n_channels, n_channels // 16)
        self.down1 = Down(n_channels // 16, n_channels // 8)
        self.down2 = Down(n_channels // 8, n_channels // 4)
        self.down3 = Down(n_channels // 4, n_channels // 2)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_channels // 2, n_channels // factor)
        self.up1 = Up((n_channels // 1) // factor, (n_channels // 2) // factor, bilinear)
        self.up2 = Up((n_channels // 2) // factor, (n_channels // 4) // factor, bilinear)
        self.up3 = Up((n_channels // 4) // factor, (n_channels // 8) // factor, bilinear)
        self.up4 = Up((n_channels // 8) // factor, (n_channels // 16) // factor, bilinear)
        self.outc = OutConv(n_channels // 16, self.n_classes)

    def forward(self, token_embeddings, attention_mask):
        x = token_embeddings
        x = rearrange(x, "batch len vocab -> batch vocab len")
        x1 = self.dropout(self.inc(x))
        x2 = self.dropout(self.down1(x1))
        x3 = self.dropout(self.down2(x2))
        x4 = self.dropout(self.down3(x3))
        x5 = self.dropout(self.down4(x4))
        x = self.dropout(self.up1(x5, x4))
        x = self.dropout(self.up2(x, x3))
        x = self.dropout(self.up3(x, x2))
        x = self.dropout(self.up4(x, x1))
        logits = self.dropout(self.outc(x))
        logits = rearrange(logits, "batch tmp len -> batch (len tmp)")

        logits[attention_mask == 0] = float("-inf")
        attn_weights = logits.softmax(dim=-1)

        output = einsum("bsv,bs->bv", token_embeddings, attn_weights)

        return output, logits
