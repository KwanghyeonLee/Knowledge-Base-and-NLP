from torch import nn
from torch.nn import functional as F

class JSDivergence(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_div = nn.KLDivLoss(reduction="none")


    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        m = 0.5 * (net_1_probs + net_1_probs)
        loss = 0.0
        loss += self.kl_div(F.log_softmax(net_1_logits, dim=1), m)
        loss += self.kl_div(F.log_softmax(net_2_logits, dim=1), m)

        return 0.5 * loss