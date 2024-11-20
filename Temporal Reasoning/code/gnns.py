from utils import replace_masked_values
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

def generate_scaled_var_drop_mask(shape, keep_prob, device='cuda'):
    assert keep_prob > 0. and keep_prob <= 1.
    mask = torch.rand(shape, device=device).le(keep_prob)
    mask = mask.float() / keep_prob
    return mask

class GCN(nn.Module):
    
    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc = torch.nn.Linear(node_dim, node_dim, bias=False)

    #def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph, extra_factor=None):
    def forward(self, d_node, q_node, qq_graph=None, dq_graph=None, dd_graph = None, qd_graph = None):
        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)
        self.qq_graph=qq_graph

        d_node_neighbor_num = dd_graph.sum(-1) + dq_graph.sum(-1) 
        # d_node_neighbor_num = dd_graph.sum(-1) + dd_graph_right.sum(-1) + dq_graph.sum(-1) + dq_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        if self.qq_graph is not None:
            q_node_neighbor_num = qq_graph.sum(-1) + qd_graph.sum(-1)
            #q_node_neighbor_num = qq_graph.sum(-1) + qq_graph_right.sum(-1) + qd_graph.sum(-1) + qd_graph_right.sum(-1)
            q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
            q_node_neighbor_num = replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)


        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            # d            
            d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)         #[bs, 92, 1024]   -> [bs, 92, 92]
            all_d_weight.append(d_node_weight)
            self_d_node_info = self._self_node_fc(d_node)
            dd_node_info = self._dd_node_fc(d_node)
            qd_node_info = self._qd_node_fc(d_node)

            dd_node_weight = replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1), # [bs, 92, 92]
                    dd_graph,
                    0)
            qd_node_weight = replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qd_graph,
                    0)
            
            dd_node_info = torch.matmul(dd_node_weight, dd_node_info) # [bs, 92, 92] * [bs, 92, node_dim]
            qd_node_info = torch.matmul(qd_node_weight, qd_node_info)
            
            # q

            q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            all_q_weight.append(q_node_weight)
            self_q_node_info = self._self_node_fc(q_node)
            qq_node_info = self._qq_node_fc(q_node)
            dq_node_info = self._dq_node_fc(q_node)

            qq_node_weight = replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qq_graph,
                    0)

            dq_node_weight = replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dq_graph,
                    0)

            qq_node_info = torch.matmul(qq_node_weight, qq_node_info)
            dq_node_info = torch.matmul(dq_node_weight, dq_node_info)

            agg_q_node_info = (qq_node_info + qd_node_info) / q_node_neighbor_num.unsqueeze(-1)
            agg_d_node_info = (dd_node_info + dq_node_info) / d_node_neighbor_num.unsqueeze(-1)
        
            d_node = F.relu(self_d_node_info + agg_d_node_info)            
            q_node = F.relu(self_q_node_info + agg_q_node_info)
    
        all_q_weight = [weight.unsqueeze(1) for weight in all_q_weight]
        all_q_weight = torch.cat(all_q_weight, dim=1)
    
        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        all_d_weight = torch.cat(all_d_weight, dim=1)
        

        return d_node, q_node, all_d_weight, all_q_weight # d_node_weight, q_node_weight
    