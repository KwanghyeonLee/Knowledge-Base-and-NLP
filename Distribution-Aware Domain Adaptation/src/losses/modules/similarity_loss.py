import torch
from einops import rearrange, repeat
from torch import Tensor, einsum, nn
from torch.nn.functional import normalize


class EmbeddingNegativeCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineSimilarity(dim=-1)

    def forward(self, single: Tensor, multi: Tensor):
        assert single.shape[0] == multi.shape[0]
        assert single.shape[-1] == multi.shape[-1]

        # this case includes pe samples
        if len(multi.shape) == 4:
            multi = rearrange(multi, "batch sample len vocab -> (batch sample) len vocab")
        # possible_max_similarity = multi.shape[0]

        single = normalize(single, p=2, dim=-1)
        multi = normalize(multi, p=2, dim=-1)

        term_similarity = self.loss_fn(
            repeat(single, "batch vocab -> batch len vocab", len=1), multi
        )
        sum_term = torch.clamp(term_similarity.sum(dim=-1), min=1e-12)  # (b,)

        batch_similarity = sum_term / term_similarity.shape[-1]  # (b,)
        sum_batch = torch.clamp(batch_similarity.sum(dim=-1), min=1e-12)  # (1,)

        output = sum_batch / term_similarity.shape[-1]
        return 1 - output
