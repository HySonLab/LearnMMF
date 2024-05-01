import pytorch_lightning as pl
from model import GNN
import torch.nn as nn
import torch
from data import MMFTrainSet
from torch.utils.data import DataLoader
from utils import get_cost
import numpy as np

class MMFAgent(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, n_layers, L, k,
                 n_batch_per_epoch, train_batch_size, matrix_size, device,
                 test_batch_size,
                 lr):
        super().__init__()
        self.net = GNN(input_dim, hidden_dim, n_layers, L, k)
        self.training_step_loss = []
        self.n_batch_per_epoch = n_batch_per_epoch
        self.train_batch_size = train_batch_size
        self.matrix_size = matrix_size
        self.device = device
        self.test_batch_size = test_batch_size
        self.lr = lr

    def forward(self, x, A):
        return self.net(x, A)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train_dataloader(self):
        train_set = MMFTrainSet(self.n_batch_per_epoch, self.train_batch_size, self.matrix_size, self.device)
        return DataLoader(train_set, batch_size=None, num_workers=0)

    def training_step(self, batch, batch_idx):
        self.net = self.net.train()
        wavelet_indices, rest_indices, log_prob = self.net(batch['x'], batch['A'])

        with torch.no_grad():
            cost = get_cost(batch['A'], self.L, self.k, wavelet_indices, rest_indices)
            baseline_cost = 0
            advatange = cost - baseline_cost

        loss = advatange * log_prob.squeeze(-1) # TODO: Check dimension here
        loss = loss.mean()
        logs = {'adv': advatange.mean().item(), 'loss': loss.item()}
        self.training_step_loss.append(torch.mean(cost).item())

        return {'loss': loss, 'log': logs}

    def on_train_epoch_end(self):
        self.log('Training loss for this epoch', np.mean(self.training_step_loss))
        self.training_step_loss = []
