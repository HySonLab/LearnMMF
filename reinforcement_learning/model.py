import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, L, k):
        super(GNN, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.k = k
        self.L = L

    def forward(self, x, A):
        sum_log_prob = 0
        agg_wavelet_indices = []
        agg_rest_indices = []
        batch_size, N, _ = A.size()
        mask = torch.zeros((batch_size, N), dtype=torch.bool, device=A.device)

        for _ in range(self.L):
            wavelet_index, rest_indices, log_prob = self._run_single_level(x=x, A=A, mask=mask)
            sum_log_prob += log_prob
            agg_wavelet_indices.append(wavelet_index.unsqueeze(-1))
            agg_rest_indices.append(rest_indices.unsqueeze(1))

        agg_wavelet_indices = torch.cat(agg_wavelet_indices, dim=-1)
        agg_rest_indices = torch.cat(agg_rest_indices, dim=1)

        return agg_wavelet_indices, agg_rest_indices, sum_log_prob

    def _run_single_level(self, x, A, mask):
        # First round of message passing
        h = torch.bmm(A, x)
        h = self.embed(h)
        h = torch.relu(h)
        h_all_layers = [h]

        # Next n_layers round
        for layer in self.layers:
            h = torch.bmm(A, h)
            h = layer(h)
            h = torch.relu(h)
            h_all_layers.append(h)

        # Concatenate all layers' output
        h_concatenated = torch.cat(h_all_layers, dim=-1)

        # Calculate the logits
        p_hat = torch.sum(h_concatenated, dim=-1)
        p_hat.masked_fill_(mask, float('-inf')) # Mask out chosen indices
        logits = F.log_softmax(p_hat, dim=-1)

        # Sample the next wavelet index
        m = Categorical(logits=logits)
        wavelet_index = m.sample()
        log_prob = m.log_prob(wavelet_index)
        mask.scatter_(dim=-1, index=wavelet_index.unsqueeze(-1), src=torch.ones_like(mask, dtype=torch.bool, device=A.device))

        # Find other k - 1 indices 
        rest_indices = self._find_most_similar_indices(h_concatenated, wavelet_index, self.k - 1)

        return wavelet_index, rest_indices, log_prob
    
    def _find_most_similar_indices(self, h, pivot, k):
        batch_size, _, d = h.size()
        pivot_row = torch.gather(h, 1, pivot.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, d))
        distances = torch.linalg.vector_norm(h - pivot_row.expand_as(h), dim=-1)
        distances.scatter_(dim=-1, index=pivot.unsqueeze(-1), src=torch.full_like(distances, float('inf'), device=h.device))
        _, topk_indices = distances.topk(k, dim=-1, largest=False)
        return topk_indices