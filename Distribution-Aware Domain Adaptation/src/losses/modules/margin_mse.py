from torch import Tensor, nn


class MarginMSELoss(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.scale = scale

    def forward(self, pred_pos: Tensor, pred_neg: Tensor, label: Tensor):
        assert pred_pos.shape == pred_neg.shape
        margin_pred = (pred_pos - pred_neg) * self.scale

        return self.loss_fn(margin_pred, label)
