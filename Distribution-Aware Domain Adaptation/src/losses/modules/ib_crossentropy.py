from torch import Tensor, nn


class InBatchCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred_in_batch: Tensor, label_in_batch: Tensor):
        predict_softmax = pred_in_batch.softmax(dim=-1)
        target_idxs = label_in_batch.softmax(dim=-1).argmax(dim=-1)

        return self.loss_fn(predict_softmax, target_idxs)
