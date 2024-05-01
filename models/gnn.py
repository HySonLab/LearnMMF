import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, L, k):
        super(GCN, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.k = k
        self.L = L

    def forward(self, x, A):
        for _ in range(self.L):
            wavelet_index, rest_indices = self.run_single_level(x, A)

    
    def run_single_level(self, x, A):
        # Fisrt round of message passing
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

        # Calculate the probabilities
        p_hat = torch.sum(h_concatenated, dim=-1)
        logits = F.log_softmax(p_hat, dim=-1)

        # Gumbel-max trick
        wavelet_index = F.gumbel_softmax(logits, tau=0.1, hard=True)

        for _ in range(self.k - 1):
            wavelet_row = torch.bmm(wavelet_index.unsqueeze(1), h_concatenated)
            similarity = torch.bmm(h_concatenated, wavelet_row.transpose(1, 2)).squeeze(-1)
            

        return wavelet_index