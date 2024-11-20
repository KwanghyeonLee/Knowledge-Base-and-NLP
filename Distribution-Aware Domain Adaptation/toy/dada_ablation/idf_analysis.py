import json
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn

dataset = "scifact"
data_path = Path("/workspace/tmp") / dataset
beir_path = Path("/workspace/beir") / dataset

q_path = data_path / "q"
d_path = data_path / "d"

idf = torch.tensor(
    json.load(
        open(beir_path / "idf" / "idfs_zero_unseen.json")
    )
)

# this measures how much the idf distribution differs from the distribution of the vocab
diff_loss = nn.KLDivLoss(reduction="batchmean")
diff = 0
diff_cnt = 0

def entropy_loss(t):
    return (-t * t.log2()).sum(dim=-1).mean()

print("uniform", entropy_loss(torch.tensor([0.5, 0.5])))
print("biased ", entropy_loss(torch.tensor([0.01, 0.01, 0.01, 0.2, 0.3, 0.3, 0.1, 0.07])))

# this measures does the distribution of the vocab is uniform or not
entropy = 0
entropy_cnt = 0

# Does the prediction of the model is almost same or not
average = torch.zeros(idf.shape)
average_cnt = 0

squared = torch.zeros(idf.shape)
squared_cnt = 0

for q_file in q_path.iterdir():
    q = torch.load(q_file)
    q_vocab = q["vocab"]
    q_vocab_prob = q_vocab.softmax(dim=-1)

    diff_loss_val = diff_loss(q_vocab_prob.log(), idf.to(q_vocab_prob.device).softmax(dim=-1))
    diff += diff_loss_val.item()
    diff_cnt += 1

    entropy_loss_val = entropy_loss(q_vocab_prob)
    entropy += entropy_loss_val.item()
    entropy_cnt += 1

    average += q_vocab_prob.mean(dim=0).to("cpu")
    average_cnt += 1

    squared += (q_vocab_prob ** 2).mean(dim=0).to("cpu")
    squared_cnt += 1

    print(".", end="")

print()
print(diff / diff_cnt)
print(entropy / entropy_cnt)
average_val = average / average_cnt
squared_val = squared / squared_cnt
variance = squared_val - average_val ** 2

plt.plot(average_val.numpy())
plt.show()

plt.plot(variance.numpy())
plt.show()

for d_file in d_path.iterdir():
    d = torch.load(d_file)
    d_vocab = d["vocab"]
    d_vocab_prob = d_vocab.softmax(dim=-1)

    diff_loss_val = diff_loss(d_vocab_prob.log(), idf.to(d_vocab_prob.device).softmax(dim=-1))
    diff += diff_loss_val.item()
    diff_cnt += 1

    entropy_loss_val = entropy_loss(d_vocab_prob)
    entropy += entropy_loss_val.item()
    entropy_cnt += 1

    average += d_vocab_prob.mean(dim=0).to("cpu")
    average_cnt += 1

    squared += (d_vocab_prob ** 2).mean(dim=0).to("cpu")
    squared_cnt += 1

    print(">", end="")

print()
print(diff / diff_cnt)
print(entropy / entropy_cnt)
average_val = average / average_cnt
squared_val = squared / squared_cnt
variance = squared_val - average_val ** 2

plt.plot(average_val.numpy())
plt.show()

plt.plot(variance.numpy())
plt.show()

print()

