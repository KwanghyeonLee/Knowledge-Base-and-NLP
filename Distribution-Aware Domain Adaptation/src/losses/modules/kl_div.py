from torch import Tensor, nn


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, pred_in_batch: Tensor, label_in_batch: Tensor):
        predict_logsoftmax = pred_in_batch.log_softmax(dim=-1)
        target_softmax = label_in_batch.softmax(dim=-1)

        return self.loss_fn(predict_logsoftmax, target_softmax)
